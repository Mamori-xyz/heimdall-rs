//! Streaming, deterministic build-trace.
//!
//! This is a *new* trace-building path that lives alongside the tree-building
//! [`VM::build_all_traces`](super) / [`VM::build_trace_start_from_route`](super) (which are left
//! untouched). Instead of materializing the whole CFG as a nested [`VMTrace`] tree in memory and
//! re-linking it bottom-up, it walks the contract once (iterative BFS, no recursion) and **streams**
//! every per-step state and every segment out through a caller-supplied [`TraceSink`]. Nothing is
//! retained globally except the BFS frontier, the loop-detection maps, and a "seen ids" set.
//!
//! Two design goals beyond streaming:
//!
//! * **Deterministic content-hash IDs.** The heavy `Arc`-shared data (`WrappedOpcode` expression
//!   trees, the stack, memory, storage, events) is replaced on the wire by a `U256` content hash.
//!   The hash is computed bottom-up, so a nested `Arc<WrappedOpcode>` resolves to its child ids
//!   first — the Arc-in-Arc case is handled, and the same value always gets the same id no matter
//!   how many times the simulation is re-run. Each interned value is emitted exactly once; a
//!   consumer rebuilds the value (and can call [`WrappedOpcode::solidify`]) from the interned
//!   records and an id.
//!
//! * **Deterministic chained segment ids.** Each segment gets a 32-byte "block hash" style id
//!   (replacing the old `u32` route-hash node id): a child reached by a JUMPI *continue*
//!   (fall-through) hashes the parent id unchanged (`delta = 0`); every other transition (JUMP
//!   target, JUMPI jump-target, or any non-jump/jumpi segment end) hashes `parent + 1`
//!   (`delta = 1`). Each emitted [`StreamSegment`] carries only its own id plus its parent and
//!   children hashes, so the CFG is rebuilt externally with no recursion / no stack overflow.
//!
//! Loop detection and the branch/segment/loop limits are **identical** to
//! [`VM::build_all_traces`](super): the same [`VM::jump_stack_hash_helper`](super) content hash and
//! visit counters bound the walk. Only the node *id* changed from `u32` to the chained `U256` hash.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ethers::core::utils::keccak256;
use ethers::prelude::U256;
use eyre::{OptionExt, Result};
use serde::{Deserialize, Serialize};

use heimdall_common::utils::strings::decode_hex;

use crate::core::{
    log::Log,
    memory::Memory,
    opcodes::{WrappedInput, WrappedOpcode},
    stack::{Stack, StackFrame},
    storage::Storage,
    vm::{Instruction, State, VM},
};

use super::{jumpi_static_outcome, static_jumpi_of_trace, StaticJumpi, VMTrace};

// Domain-separation tags so an opcode id can never collide with a stack/memory/storage/etc. id.
const TAG_OPCODE: u8 = 0x01;
const TAG_FRAME: u8 = 0x02;
const TAG_STACK: u8 = 0x03;
const TAG_MEMORY: u8 = 0x04;
const TAG_STORAGE: u8 = 0x05;
const TAG_EVENTS: u8 = 0x06;

/// Genesis id for the root segment of any walk. A fixed constant so a fresh full walk and an
/// extend/resume of the same contract agree on the root id (and therefore on every chained id
/// derived from it) regardless of how the node was reached.
fn genesis() -> U256 {
    U256::from(keccak256(b"heimdall::stream::genesis::v1"))
}

/// Canonical 32-byte big-endian encoding of a [`U256`], used everywhere we feed a `U256` into a
/// keccak preimage so the encoding is stable across runs and platforms.
fn u256_be(v: U256) -> [u8; 32] {
    let mut b = [0u8; 32];
    v.to_big_endian(&mut b);
    b
}

/// Chained "block hash" id of a child segment, per the scheme described in the module docs.
/// `delta = 0` => continue (hash the parent id unchanged); `delta = 1` => hash `parent + 1`.
/// The `+ 1` wraps on overflow (only reachable for an astronomically unlikely `U256::MAX` parent).
fn chained_segment_hash(parent: U256, delta: u8) -> U256 {
    let seed = if delta == 0 { parent } else { parent.overflowing_add(U256::one()).0 };
    U256::from(keccak256(u256_be(seed)))
}

/// How a child segment was reached from its parent. Drives the chained-hash `delta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchKind {
    /// JUMPI fall-through (`delta = 0`).
    Continue,
    /// JUMP target, JUMPI jump-target, or any non-jump/jumpi segment end (`delta = 1`).
    Jump,
}

impl BranchKind {
    fn delta(self) -> u8 {
        match self {
            BranchKind::Continue => 0,
            BranchKind::Jump => 1,
        }
    }
}

/// Classify a successor by comparing its start instruction against the segment's last instruction.
/// Only a JUMPI fall-through is a [`BranchKind::Continue`]; everything else is a [`BranchKind::Jump`].
fn successor_branch_kind(last: &Instruction, child_instruction: u128) -> BranchKind {
    if last.opcode == 0x57 && child_instruction == last.instruction + 1 {
        BranchKind::Continue
    } else {
        BranchKind::Jump
    }
}

/// A reference inside an interned [`WrappedOpcode`]: either a raw literal or another interned opcode
/// (identified by its content-hash id). Mirrors [`WrappedInput`] with the `Arc<WrappedOpcode>`
/// replaced by its id so the consumer can rebuild the tree bottom-up.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputRef {
    Raw(U256),
    /// Content-hash id of the nested interned opcode.
    Op(U256),
    /// Memory-read value provenance: one `(start, end, op-id)` per covering write (`end` exclusive,
    /// op-id is the content-hash of the producing write op). Mirrors `WrappedInput::MemorySlice`.
    MemorySlice(Vec<(u64, u64, U256)>),
    /// Concrete `SHA3`/`KECCAK256` result carried in the op's identity. Mirrors `WrappedInput::KeccakResult`.
    KeccakResult(U256),
}

/// One executed instruction's state, with every heavy field replaced by an interned id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamStep {
    /// Chained id of the segment this step belongs to.
    pub segment_hash: U256,
    /// Ordinal of this step within its segment (0-based).
    pub step_index: u32,
    pub instruction_pc: u128,
    pub opcode: u8,
    pub gas_used: u128,
    pub gas_remaining: u128,
    /// Concrete values popped as inputs (kept inline — cheap).
    pub inputs: Vec<U256>,
    /// Concrete values pushed as outputs.
    pub outputs: Vec<U256>,
    /// Interned ids of the symbolic input expressions (`input_operations`).
    pub input_operation_ids: Vec<U256>,
    /// Interned ids of the symbolic output expressions (`output_operations`).
    pub output_operation_ids: Vec<U256>,
    pub stack_id: U256,
    pub memory_id: U256,
    pub storage_id: U256,
    pub events_id: U256,
}

/// An edge from a segment to one of its successors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildEdge {
    /// Chained id of the successor segment.
    pub chained_hash: U256,
    /// Loop-detection (content) hash of the successor's first instruction + stack — the same hash
    /// [`VM::build_all_traces`](super) uses, equal to the successor's own `loop_hash` once emitted.
    pub loop_hash: U256,
    pub branch_kind: BranchKind,
}

/// A finished segment. Carries only its own id plus parent/children hashes so the CFG can be
/// rebuilt externally without recursion. A child listed here that is pruned by the loop limit
/// simply never emits its own segment (a bounded frontier edge).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSegment {
    /// Deterministic 32-byte chained id (plays the role of the old `u32` node id).
    pub segment_hash: U256,
    /// Chained id of the parent segment; `None` for the walk/extend root.
    pub parent_hash: Option<U256>,
    /// Intended successors (materialized iff their own [`StreamSegment`] is later emitted).
    pub children: Vec<ChildEdge>,
    /// Loop-detection (content) hash for this segment — `== VMTraceExtended.hash` in the old API.
    pub loop_hash: U256,
    /// Loop hashes of the immediately-reachable successors, computed by the same formula as
    /// `build_all_traces`' `next_possible_segment_hashes` (kept byte-identical to the tree engine,
    /// including its quirk of listing a fall-through hash for terminal opcodes). Distinct from
    /// `children`, which lists only successors that are actually explored.
    pub next_possible_segment_hashes: HashSet<U256>,
    /// PC of the segment's first instruction.
    pub start_pc: u128,
    /// PC of the segment's last (terminating) instruction.
    pub end_pc: u128,
    pub step_count: u32,
    /// Set when the segment ends in a statically-resolved JUMPI (reuses `static_jumpi_of_trace`).
    pub static_jumpi: Option<StaticJumpi>,
}

/// Outcome of a streaming walk. All data was already streamed through the sink; this reports how
/// much was explored and whether the walk drained naturally or stopped at a limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSummary {
    /// `true` if the BFS queue drained; `false` if a `simple_cfg` branch/segment limit broke it.
    pub completed: bool,
    pub segment_count: u32,
    pub branch_count: u32,
}

/// Sink that receives streamed trace data. Every method has a default no-op body, so a consumer
/// implements only what it needs (e.g. ignore `intern_memory` if it never inspects memory).
///
/// Ordering guarantees within a walk:
/// * `intern_*` for a value is emitted before any `StreamStep` that references its id.
/// * For an interned opcode, all of its child opcodes are interned before it (bottom-up), so a
///   consumer can rebuild expression trees in arrival order.
/// * A segment's `on_step`s are emitted before its `on_segment`; segments arrive in BFS order
///   (parents before children).
pub trait TraceSink {
    /// A newly-seen `WrappedOpcode`: `code` is the opcode byte, `inputs` its operands.
    fn intern_opcode(&mut self, _id: U256, _code: u8, _inputs: &[InputRef]) {}
    /// A newly-seen stack frame (`value` + the id of the `WrappedOpcode` that produced it).
    fn intern_stack_frame(&mut self, _id: U256, _value: U256, _op_id: U256) {}
    /// A newly-seen full stack, as the ordered list of its frame ids (bottom to top).
    fn intern_stack(&mut self, _id: U256, _frame_ids: &[U256]) {}
    /// A newly-seen memory image.
    fn intern_memory(&mut self, _id: U256, _bytes: &[u8]) {}
    /// Byte-tracker provenance for the memory image just interned (same `id`): the ranges that make
    /// up the image, each as `(start, end_inclusive, op_id)` where `op_id` is the interned id of the
    /// `WrappedOpcode` that last wrote that range (its operands are interned before this call, like
    /// any other opcode). Sorted by `(start, end)`. Emitted right after [`Self::intern_memory`] for
    /// the same `id`, and only when the `experimental` byte-tracker is compiled in — otherwise never
    /// called. Lets a consumer rebuild `Memory.bytes` (the provenance map that drives mamori's
    /// `memory_raw_values`), which the flat byte image alone cannot reconstruct.
    fn intern_memory_provenance(&mut self, _id: U256, _ranges: &[(u64, u64, U256)]) {}
    /// A newly-seen storage image: persistent `entries` and `transient`, each sorted by key.
    fn intern_storage(&mut self, _id: U256, _entries: &[(U256, U256)], _transient: &[(U256, U256)]) {}
    /// A newly-seen events log.
    fn intern_events(&mut self, _id: U256, _logs: &[Log]) {}
    /// The concrete result of an executed `SHA3` (`KECCAK256`), keyed by the interned id of the SHA3
    /// `WrappedOpcode` itself (`op_id` == that step's sole `output_operation_id`). Emitted once, at the
    /// step where the SHA3 executes. Because the id is the SHA3 expression's content hash, the SAME id
    /// reappears nested in any later instruction whose stack operand carries that SHA3 (e.g. a mapping
    /// `SLOAD(SHA3(..))` flowing across segments) — so a consumer can recover the keccak slot value by
    /// id alone, without replaying the SHA3 (whose preimage memory may already be overwritten).
    fn intern_keccak_eval(&mut self, _op_id: U256, _result: U256) {}

    /// One executed instruction's state.
    fn on_step(&mut self, _step: &StreamStep) {}
    /// A finished segment.
    fn on_segment(&mut self, _segment: &StreamSegment) {}
}

/// Content-hash interner. Produces order-independent `U256` ids and emits each value through the
/// sink exactly once. Within a single run it memoizes by `Arc` pointer identity so the common case
/// (the same shared `Arc` reappearing across steps) skips re-hashing — this is what preserves the
/// `Arc` speed-up while still producing deterministic content ids across runs.
///
/// IMPORTANT: each memo entry stores a *clone of the `Arc`* alongside the id. A bare
/// `Arc::as_ptr` key would be unsound: once an `Arc` is dropped its heap address can be reused by a
/// different value, and the stale cache entry would then hand back the wrong (previous) content id.
/// Holding the `Arc` pins the allocation, so a given address maps to exactly one value for the
/// interner's lifetime — making a pointer hit provably the same content. Entries are bounded by the
/// number of *distinct* interned values (already deduped), not by step count.
#[derive(Default)]
struct Interner {
    /// Ids already streamed via an `intern_*` call (dedup across all value types; ids are
    /// domain-tagged so cross-type collisions are impossible).
    emitted: HashSet<U256>,
    op_ptr: HashMap<usize, (Arc<WrappedOpcode>, U256)>,
    frame_ptr: HashMap<usize, (Arc<StackFrame>, U256)>,
    mem_ptr: HashMap<usize, (Arc<Memory>, U256)>,
    storage_ptr: HashMap<usize, (Arc<Storage>, U256)>,
    events_ptr: HashMap<usize, (Arc<Vec<Log>>, U256)>,
}

impl Interner {
    fn intern_opcode<S: TraceSink>(&mut self, sink: &mut S, op: &WrappedOpcode) -> U256 {
        let mut refs: Vec<InputRef> = Vec::with_capacity(op.inputs.len());
        for input in &op.inputs {
            match input {
                WrappedInput::Raw(v) => refs.push(InputRef::Raw(*v)),
                WrappedInput::Opcode(arc) => refs.push(InputRef::Op(self.intern_opcode_arc(sink, arc))),
                WrappedInput::MemorySlice(segments) => {
                    let segs = segments
                        .iter()
                        .map(|s| (s.start as u64, s.end as u64, self.intern_opcode_arc(sink, &s.op)))
                        .collect();
                    refs.push(InputRef::MemorySlice(segs));
                }
                WrappedInput::KeccakResult(v) => refs.push(InputRef::KeccakResult(*v)),
            }
        }
        let id = hash_opcode(op.opcode.code, &refs);
        if self.emitted.insert(id) {
            sink.intern_opcode(id, op.opcode.code, &refs);
        }
        id
    }

    fn intern_opcode_arc<S: TraceSink>(&mut self, sink: &mut S, arc: &Arc<WrappedOpcode>) -> U256 {
        let ptr = Arc::as_ptr(arc) as usize;
        if let Some((_, id)) = self.op_ptr.get(&ptr) {
            return *id;
        }
        let id = self.intern_opcode(sink, arc.as_ref());
        self.op_ptr.insert(ptr, (arc.clone(), id));
        id
    }

    fn intern_frame<S: TraceSink>(&mut self, sink: &mut S, frame: &Arc<StackFrame>) -> U256 {
        let ptr = Arc::as_ptr(frame) as usize;
        if let Some((_, id)) = self.frame_ptr.get(&ptr) {
            return *id;
        }
        let op_id = self.intern_opcode(sink, &frame.operation);
        let mut data = Vec::with_capacity(1 + 64);
        data.push(TAG_FRAME);
        data.extend_from_slice(&u256_be(frame.value));
        data.extend_from_slice(&u256_be(op_id));
        let id = U256::from(keccak256(&data));
        if self.emitted.insert(id) {
            sink.intern_stack_frame(id, frame.value, op_id);
        }
        self.frame_ptr.insert(ptr, (frame.clone(), id));
        id
    }

    fn intern_stack<S: TraceSink>(&mut self, sink: &mut S, stack: &Stack) -> U256 {
        let mut frame_ids: Vec<U256> = Vec::with_capacity(stack.stack.len());
        for frame in &stack.stack {
            frame_ids.push(self.intern_frame(sink, frame));
        }
        let mut data = Vec::with_capacity(1 + frame_ids.len() * 32);
        data.push(TAG_STACK);
        for fid in &frame_ids {
            data.extend_from_slice(&u256_be(*fid));
        }
        let id = U256::from(keccak256(&data));
        if self.emitted.insert(id) {
            sink.intern_stack(id, &frame_ids);
        }
        id
    }

    fn intern_memory<S: TraceSink>(&mut self, sink: &mut S, mem: &Arc<Memory>) -> U256 {
        let ptr = Arc::as_ptr(mem) as usize;
        if let Some((_, id)) = self.mem_ptr.get(&ptr) {
            return *id;
        }
        // Byte-tracker provenance (experimental only): intern each writing opcode bottom-up (so its
        // id is emitted before this memory references it) and fold the sorted ranges into the
        // content id. Without this, two images with identical bytes but different provenance would
        // alias to one id and the second's provenance would be lost.
        #[cfg(feature = "experimental")]
        let provenance: Vec<(u64, u64, U256)> = {
            let mut v: Vec<(u64, u64, U256)> = Vec::with_capacity(mem.bytes.0.len());
            for (range, op) in mem.bytes.0.iter() {
                let op_id = self.intern_opcode(sink, op);
                v.push((range.start as u64, range.end as u64, op_id));
            }
            v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            v
        };

        let mut data = Vec::with_capacity(1 + mem.memory.len());
        data.push(TAG_MEMORY);
        data.extend_from_slice(&mem.memory);
        #[cfg(feature = "experimental")]
        {
            data.extend_from_slice(&(provenance.len() as u64).to_be_bytes());
            for (start, end, op_id) in &provenance {
                data.extend_from_slice(&start.to_be_bytes());
                data.extend_from_slice(&end.to_be_bytes());
                data.extend_from_slice(&u256_be(*op_id));
            }
        }
        let id = U256::from(keccak256(&data));
        if self.emitted.insert(id) {
            sink.intern_memory(id, &mem.memory);
            #[cfg(feature = "experimental")]
            sink.intern_memory_provenance(id, &provenance);
        }
        self.mem_ptr.insert(ptr, (mem.clone(), id));
        id
    }

    fn intern_storage<S: TraceSink>(&mut self, sink: &mut S, storage: &Arc<Storage>) -> U256 {
        let ptr = Arc::as_ptr(storage) as usize;
        if let Some((_, id)) = self.storage_ptr.get(&ptr) {
            return *id;
        }
        // Sort by key so the id is independent of HashMap iteration order.
        let mut entries: Vec<(U256, U256)> =
            storage.storage.iter().map(|(k, v)| (U256::from(*k), U256::from(*v))).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut transient: Vec<(U256, U256)> =
            storage.transient.iter().map(|(k, v)| (U256::from(*k), U256::from(*v))).collect();
        transient.sort_by(|a, b| a.0.cmp(&b.0));

        // Length-prefix each map (counts are fixed-width) so the preimage is unambiguous: a
        // separator byte would be unsafe because storage keys/values are arbitrary 32-byte words
        // and could contain it, letting a different (entries, transient) split alias.
        let mut data = Vec::with_capacity(1 + 16 + (entries.len() + transient.len()) * 64);
        data.push(TAG_STORAGE);
        data.extend_from_slice(&(entries.len() as u64).to_be_bytes());
        for (k, v) in &entries {
            data.extend_from_slice(&u256_be(*k));
            data.extend_from_slice(&u256_be(*v));
        }
        data.extend_from_slice(&(transient.len() as u64).to_be_bytes());
        for (k, v) in &transient {
            data.extend_from_slice(&u256_be(*k));
            data.extend_from_slice(&u256_be(*v));
        }
        let id = U256::from(keccak256(&data));
        if self.emitted.insert(id) {
            sink.intern_storage(id, &entries, &transient);
        }
        self.storage_ptr.insert(ptr, (storage.clone(), id));
        id
    }

    fn intern_events<S: TraceSink>(&mut self, sink: &mut S, events: &Arc<Vec<Log>>) -> U256 {
        let ptr = Arc::as_ptr(events) as usize;
        if let Some((_, id)) = self.events_ptr.get(&ptr) {
            return *id;
        }
        // Length-prefix the log list and each log's variable-length fields (topics, data) so the
        // preimage is unambiguous — `log.data` is arbitrary bytes and could contain any separator.
        let mut data = Vec::new();
        data.push(TAG_EVENTS);
        data.extend_from_slice(&(events.len() as u64).to_be_bytes());
        for log in events.iter() {
            data.extend_from_slice(&log.index.to_be_bytes());
            data.extend_from_slice(&(log.topics.len() as u64).to_be_bytes());
            for topic in &log.topics {
                data.extend_from_slice(&u256_be(*topic));
            }
            data.extend_from_slice(&(log.data.len() as u64).to_be_bytes());
            data.extend_from_slice(&log.data);
        }
        let id = U256::from(keccak256(&data));
        if self.emitted.insert(id) {
            sink.intern_events(id, events.as_slice());
        }
        self.events_ptr.insert(ptr, (events.clone(), id));
        id
    }

    /// Intern every heavy field of a [`State`] and build the flat [`StreamStep`] for it.
    fn step_record<S: TraceSink>(
        &mut self,
        sink: &mut S,
        segment_hash: U256,
        step_index: u32,
        state: &State,
    ) -> StreamStep {
        let input_operation_ids = state
            .last_instruction
            .input_operations
            .iter()
            .map(|op| self.intern_opcode(sink, op))
            .collect();
        let output_operation_ids: Vec<U256> = state
            .last_instruction
            .output_operations
            .iter()
            .map(|op| self.intern_opcode(sink, op))
            .collect();
        // SHA3 (KECCAK256, 0x20): record its concrete result keyed by the SHA3 expression's id (its
        // sole output operation). Lets a later segment recover a mapping/array slot it carries on the
        // stack without re-running the SHA3 — see `TraceSink::intern_keccak_eval`.
        if state.last_instruction.opcode == 0x20 {
            if let (Some(&op_id), Some(&result)) =
                (output_operation_ids.first(), state.last_instruction.outputs.first())
            {
                sink.intern_keccak_eval(op_id, result);
            }
        }
        let stack_id = self.intern_stack(sink, &state.stack);
        let memory_id = self.intern_memory(sink, &state.memory);
        let storage_id = self.intern_storage(sink, &state.storage);
        let events_id = self.intern_events(sink, &state.events);

        StreamStep {
            segment_hash,
            step_index,
            instruction_pc: state.last_instruction.instruction,
            opcode: state.last_instruction.opcode,
            gas_used: state.gas_used,
            gas_remaining: state.gas_remaining,
            inputs: state.last_instruction.inputs.clone(),
            outputs: state.last_instruction.outputs.clone(),
            input_operation_ids,
            output_operation_ids,
            stack_id,
            memory_id,
            storage_id,
            events_id,
        }
    }
}

/// Content-hash of a `WrappedOpcode` from its code and (already-resolved) input refs.
fn hash_opcode(code: u8, refs: &[InputRef]) -> U256 {
    let mut data: Vec<u8> = Vec::with_capacity(2 + refs.len() * 33);
    data.push(TAG_OPCODE);
    data.push(code);
    for r in refs {
        match r {
            InputRef::Raw(v) => {
                data.push(0);
                data.extend_from_slice(&u256_be(*v));
            }
            InputRef::Op(id) => {
                data.push(1);
                data.extend_from_slice(&u256_be(*id));
            }
            InputRef::MemorySlice(segs) => {
                data.push(2);
                data.extend_from_slice(&(segs.len() as u64).to_be_bytes());
                for (s, e, id) in segs {
                    data.extend_from_slice(&s.to_be_bytes());
                    data.extend_from_slice(&e.to_be_bytes());
                    data.extend_from_slice(&u256_be(*id));
                }
            }
            InputRef::KeccakResult(v) => {
                data.push(3);
                data.extend_from_slice(&u256_be(*v));
            }
        }
    }
    U256::from(keccak256(&data))
}

/// Emit one finished segment: stream its steps, then its segment record. Returns the
/// `(child_chained_id, child_vm)` pairs the caller should enqueue (all successors — the caller
/// decides whether to actually enqueue based on the loop limit).
#[allow(clippy::too_many_arguments)]
/// Loop hashes of a finished segment's immediately-reachable successors. Replicates
/// `build_all_traces`' `next_possible_segment_hashes_fn` exactly so the streaming and tree engines
/// produce identical `next_possible_segment_hashes` (including its terminal-opcode fall-through quirk).
fn next_possible_segment_hashes(trace: &VMTrace, jumpdest_pc: &HashSet<U256>) -> HashSet<U256> {
    let mut hashes = HashSet::new();
    let op = match trace.operations.last() {
        Some(op) => op,
        None => return hashes,
    };
    let opcode = op.last_instruction.opcode as u128;
    let last_instruction = op.last_instruction.instruction;
    match opcode {
        0x57_u128 => {
            let jump_target = op.last_instruction.inputs[0].as_u128() + 1;
            let fall_through = last_instruction + 1;
            match jumpi_static_outcome(&op.last_instruction) {
                Some(true) => {
                    hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, jump_target, &op.stack));
                }
                Some(false) => {
                    hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, fall_through, &op.stack));
                }
                None => {
                    hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, jump_target, &op.stack));
                    hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, fall_through, &op.stack));
                }
            }
        }
        0x56_u128 => {
            hashes.insert(VM::jump_stack_hash_helper(
                jumpdest_pc,
                op.last_instruction.inputs[0].as_u128() + 1,
                &op.stack,
            ));
        }
        0x5f_u128..=0x7f_u128 => {
            let next_instruction = last_instruction + (opcode - 0x5f) + 1;
            hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, next_instruction, &op.stack));
        }
        _ => {
            hashes.insert(VM::jump_stack_hash_helper(jumpdest_pc, last_instruction + 1, &op.stack));
        }
    }
    hashes
}

fn emit_node<S: TraceSink>(
    sink: &mut S,
    interner: &mut Interner,
    jumpdest_pc: &HashSet<U256>,
    parent_id: Option<U256>,
    this_id: U256,
    this_loop_hash: U256,
    trace: &VMTrace,
    next_vms: Vec<VM>,
) -> Result<Vec<(U256, VM)>> {
    let first_pc = trace
        .operations
        .first()
        .map(|s| s.last_instruction.instruction)
        .unwrap_or(0);
    let last = trace
        .operations
        .last()
        .ok_or_eyre("cannot emit an empty segment")?
        .last_instruction
        .clone();
    let step_count = trace.operations.len() as u32;

    for (i, state) in trace.operations.iter().enumerate() {
        let step = interner.step_record(sink, this_id, i as u32, state);
        sink.on_step(&step);
    }

    let mut children = Vec::with_capacity(next_vms.len());
    let mut enqueue = Vec::with_capacity(next_vms.len());
    for child_vm in next_vms {
        let kind = successor_branch_kind(&last, child_vm.instruction);
        let child_id = chained_segment_hash(this_id, kind.delta());
        let loop_hash = VM::jump_stack_hash_helper(jumpdest_pc, child_vm.instruction, &child_vm.stack);
        children.push(ChildEdge { chained_hash: child_id, loop_hash, branch_kind: kind });
        enqueue.push((child_id, child_vm));
    }

    sink.on_segment(&StreamSegment {
        segment_hash: this_id,
        parent_hash: parent_id,
        children,
        loop_hash: this_loop_hash,
        next_possible_segment_hashes: next_possible_segment_hashes(trace, jumpdest_pc),
        start_pc: first_pc,
        end_pc: last.instruction,
        step_count,
        static_jumpi: static_jumpi_of_trace(trace),
    });

    Ok(enqueue)
}

impl VM {
    /// Streaming counterpart of [`VM::build_all_traces_selector`](super): step to the function
    /// entry point, then run [`VM::stream_all_traces`].
    #[allow(clippy::too_many_arguments)]
    pub fn stream_all_traces_selector<S: TraceSink>(
        &mut self,
        selector: &str,
        entry_point: u128,
        branch_limit: Option<u32>,
        segment_limit: Option<u32>,
        loop_limit: Option<u32>,
        simple_cfg: bool,
        route: Option<Vec<(usize, usize)>>,
        processed_nodes: &mut HashSet<U256>,
        sink: &mut S,
    ) -> Result<Option<StreamSummary>> {
        self.calldata = decode_hex(selector)?;

        while self.bytecode.len() >= self.instruction as usize && (self.instruction <= entry_point) {
            self.step()?;
            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        self.stream_all_traces(
            branch_limit,
            segment_limit,
            loop_limit,
            simple_cfg,
            route,
            processed_nodes,
            sink,
        )
    }

    /// Streaming counterpart of [`VM::build_all_traces`](super).
    ///
    /// Mirrors that function's loop detection and branch/segment/loop limits exactly. The only
    /// structural differences: it never accumulates the tree (each segment is emitted through
    /// `sink` as soon as it is built), and node ids are the chained `U256` hash instead of a `u32`
    /// counter.
    ///
    /// * `route` — an extend: replay this route to the frontier, then expand. The route-end node's
    ///   chained id is recomputed by folding the same delta rule along the route.
    /// Returns `Ok(None)` if a non-`simple_cfg` branch/segment limit aborted the walk (matching
    /// `build_all_traces`); otherwise `Ok(Some(summary))`. Note: in streaming mode, any segments
    /// explored before the limit have already been emitted through `sink`.
    #[allow(clippy::too_many_arguments)]
    pub fn stream_all_traces<S: TraceSink>(
        &mut self,
        branch_limit: Option<u32>,
        segment_limit: Option<u32>,
        loop_limit: Option<u32>,
        simple_cfg: bool,
        route: Option<Vec<(usize, usize)>>,
        processed_nodes: &mut HashSet<U256>,
        sink: &mut S,
    ) -> Result<Option<StreamSummary>> {
        let mut branch_count: u32 = 0;
        let mut segment_count: u32 = 0;

        let jumpdest_pc = VM::program_counter(self.bytecode.clone())
            .iter()
            .filter(|(_, v)| v.code == 0x5b)
            .map(|(k, _)| *k)
            .collect::<HashSet<U256>>();

        let mut interner = Interner::default();

        // Loop-detection hash of the root, computed at the current instruction *before* any
        // build/replay — identical to `build_all_traces`'s `root_trace_hash`.
        let root_loop_hash = VM::jump_stack_hash_helper(&jumpdest_pc, self.instruction, &self.stack);

        // Build the root segment + its successor VMs, and determine the root's chained id.
        let (root_trace, root_next_vms, route_hashes, root_id) = if let Some(route) = route {
            let (t, v, rh, root_id) =
                self.build_trace_start_from_route_chained(route, &jumpdest_pc)?;
            (t, v, rh, root_id)
        } else {
            let (t, v) = self.build_trace()?;
            (t, v, HashMap::new(), genesis())
        };

        // Pre-seed loop detection so the root (and any route segments) are counted.
        let mut previous_trace_hash = route_hashes;
        *previous_trace_hash.entry(root_loop_hash).or_insert(0) += 1;
        let global_limit = loop_limit.unwrap_or(4);
        let route_max_count = previous_trace_hash.values().max().copied().unwrap_or(0);

        branch_count += 1;
        segment_count += 1;

        // Emit the root, then seed the queue with its successors.
        let root_enqueue = emit_node(
            sink,
            &mut interner,
            &jumpdest_pc,
            None,
            root_id,
            root_loop_hash,
            &root_trace,
            root_next_vms,
        )?;

        let mut queue: VecDeque<(U256, U256, HashMap<U256, u32>, VM)> = VecDeque::new();
        for (child_id, child_vm) in root_enqueue {
            queue.push_back((root_id, child_id, previous_trace_hash.clone(), child_vm));
        }

        let mut completed = true;
        while !queue.is_empty() {
            if branch_limit.is_some_and(|l| branch_count >= l) {
                if simple_cfg {
                    completed = false;
                    break;
                }
                return Ok(None);
            }
            if segment_limit.is_some_and(|l| segment_count >= l) {
                if simple_cfg {
                    completed = false;
                    break;
                }
                return Ok(None);
            }

            let (parent_id, this_id, mut previous_trace_hash, mut vm) =
                queue.pop_front().ok_or_eyre("no next traces")?;
            // Loop hash of this node's segment start (== current_trace_hash in build_all_traces).
            let current_trace_hash =
                VM::jump_stack_hash_helper(&jumpdest_pc, vm.instruction, &vm.stack);
            let (trace, next_vms) = vm.build_trace()?;

            // Loop detection — identical to build_all_traces.
            let updated_count = {
                let count = previous_trace_hash.entry(current_trace_hash).or_insert(0);
                *count += 1;
                *count
            };
            let can_produce = updated_count <= route_max_count + 1;
            let skip_children = updated_count > global_limit;
            if !can_produce {
                continue;
            }

            if simple_cfg {
                let is_known_loop_segment = updated_count > 1;
                if !processed_nodes.contains(&current_trace_hash) {
                    processed_nodes.insert(current_trace_hash);
                } else if parent_id == root_id || is_known_loop_segment {
                    // allow: direct child of root, or known loop segment within loop_limit
                } else {
                    continue;
                }
            }

            let n_succ = next_vms.len();
            if n_succ > 1 {
                branch_count += 1;
            }
            segment_count += 1;

            let enqueue = emit_node(
                sink,
                &mut interner,
                &jumpdest_pc,
                Some(parent_id),
                this_id,
                current_trace_hash,
                &trace,
                next_vms,
            )?;

            if !skip_children {
                for (child_id, child_vm) in enqueue {
                    queue.push_back((this_id, child_id, previous_trace_hash.clone(), child_vm));
                }
            }
        }

        Ok(Some(StreamSummary { completed, segment_count, branch_count }))
    }

    /// Streaming-aware copy of [`VM::build_trace_start_from_route`](super) (the original is left
    /// untouched). In addition to replaying the route and returning the route-segment loop hashes,
    /// it folds the chained id along the route so the route-end node gets the same id a fresh full
    /// walk would assign. Returns `(root_trace, next_vms, route_segment_hashes, route_end_id)`.
    fn build_trace_start_from_route_chained(
        &mut self,
        mut route: Vec<(usize, usize)>,
        jumpdest_pc: &HashSet<U256>,
    ) -> Result<(VMTrace, Vec<VM>, HashMap<U256, u32>, U256)> {
        let mut root_trace = VMTrace {
            instruction: self.instruction,
            gas_used: 0,
            operations: Vec::new(),
            children: Vec::new(),
        };

        route.reverse();
        let mut current_node = route.pop().ok_or_eyre("no route")?;
        if current_node.0 != self.instruction as usize - 1 {
            return Err(eyre::eyre!("route does not start at the current instruction [1]"));
        }

        // genesis == id of route[0] (the root); each hop folds in one delta below.
        let mut chained = genesis();
        let mut next_traces = Vec::new();
        let mut route_segment_hashes: HashMap<U256, u32> = HashMap::new();

        while self.bytecode.len() >= self.instruction as usize {
            let state = self.step()?;
            let last_instruction = state.last_instruction.clone();

            if last_instruction.opcode == 0x57 {
                // jumpi
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        match jumpi_static_outcome(&last_instruction) {
                            Some(true) => {
                                self.instruction = last_instruction.inputs[0].as_u128() + 1;
                                next_traces.push(self.clone());
                            }
                            Some(false) => {
                                self.instruction = last_instruction.instruction + 1;
                                next_traces.push(self.clone());
                            }
                            None => {
                                self.instruction = last_instruction.instruction + 1;
                                next_traces.push(self.clone());
                                self.instruction = last_instruction.inputs[0].as_u128() + 1;
                                next_traces.push(self.clone());
                            }
                        }
                        break;
                    } else {
                        current_node = route.pop().ok_or_eyre("no route")?;
                        if current_node.0 == last_instruction.instruction as usize {
                            // fall-through / continue => delta 0
                            self.instruction = last_instruction.instruction + 1;
                            chained = chained_segment_hash(chained, 0);
                        } else if current_node.0 == last_instruction.inputs[0].as_u128() as usize {
                            // jump target => delta 1
                            self.instruction = last_instruction.inputs[0].as_u128() + 1;
                            chained = chained_segment_hash(chained, 1);
                        } else {
                            return Err(eyre::eyre!("route does not match the last instruction [2]"));
                        }
                        let seg_hash =
                            VM::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        *route_segment_hashes.entry(seg_hash).or_insert(0) += 1;
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [3]"));
                }
            } else if last_instruction.opcode == 0x56 {
                // jump
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        self.instruction = last_instruction.inputs[0].as_u128() + 1;
                        next_traces.push(self.clone());
                        break;
                    } else {
                        current_node = route.pop().ok_or_eyre("no route")?;
                        if current_node.0 == last_instruction.inputs[0].as_u128() as usize {
                            self.instruction = last_instruction.inputs[0].as_u128() + 1;
                            chained = chained_segment_hash(chained, 1); // jump => delta 1
                        } else {
                            return Err(eyre::eyre!("route does not match the last instruction [4]"));
                        }
                        let seg_hash =
                            VM::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        *route_segment_hashes.entry(seg_hash).or_insert(0) += 1;
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [5]"));
                }
            } else if last_instruction.opcode == 0x00
                || last_instruction.opcode == 0xfd
                || last_instruction.opcode == 0xfe
                || last_instruction.opcode == 0xff
                || last_instruction.opcode == 0xf3
            {
                // stop, revert, invalid, selfdestruct, or return: the trace ends here.
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        break;
                    } else {
                        return Err(eyre::eyre!(
                            "route should end here, but got more instructions in the route"
                        ));
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [8]"));
                }
            } else if self
                .bytecode
                .get((self.instruction - 1) as usize)
                .ok_or_eyre(format!("invalid jumpdest: {}", self.instruction - 1))?
                .to_owned()
                == 0x5b
            {
                // next instruction is a jumpdest (natural fall-through into a new segment).
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        next_traces.push(self.clone());
                        break;
                    } else {
                        current_node = route.pop().ok_or_eyre("no route")?;
                        if current_node.0 != self.instruction as usize - 1 {
                            return Err(eyre::eyre!("route does not match the last instruction [6]"));
                        }
                        // non-jump/jumpi segment end => delta 1 (treated like a jump).
                        chained = chained_segment_hash(chained, 1);
                        let seg_hash =
                            VM::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        *route_segment_hashes.entry(seg_hash).or_insert(0) += 1;
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [7]"));
                }
            }

            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        if !route.is_empty() {
            return Err(eyre::eyre!("not all routes were taken"));
        }

        Ok((root_trace, next_traces, route_segment_hashes, chained))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vm::VM;
    use ethers::types::H160;

    /// A sink that records everything for assertions, and rebuilds interned opcodes so tests can
    /// call `solidify()` on reconstructed expression trees.
    #[derive(Default)]
    struct CollectingSink {
        opcodes: HashMap<U256, (u8, Vec<InputRef>)>,
        steps: Vec<StreamStep>,
        segments: Vec<StreamSegment>,
        /// Order in which interned opcode ids were first emitted (to assert children-before-parents).
        opcode_emit_order: Vec<U256>,
    }

    impl TraceSink for CollectingSink {
        fn intern_opcode(&mut self, id: U256, code: u8, inputs: &[InputRef]) {
            self.opcodes.insert(id, (code, inputs.to_vec()));
            self.opcode_emit_order.push(id);
        }
        fn on_step(&mut self, step: &StreamStep) {
            self.steps.push(step.clone());
        }
        fn on_segment(&mut self, segment: &StreamSegment) {
            self.segments.push(segment.clone());
        }
    }

    impl CollectingSink {
        /// Rebuild a `WrappedOpcode` from the interned table by id (bottom-up).
        fn rebuild(&self, id: U256) -> WrappedOpcode {
            let (code, inputs) = self.opcodes.get(&id).expect("opcode id not interned");
            let rebuilt_inputs = inputs
                .iter()
                .map(|r| match r {
                    InputRef::Raw(v) => WrappedInput::Raw(*v),
                    InputRef::Op(child) => WrappedInput::Opcode(Arc::new(self.rebuild(*child))),
                    InputRef::MemorySlice(segs) => WrappedInput::MemorySlice(
                        segs.iter()
                            .map(|(s, e, child)| crate::core::opcodes::MemorySegment {
                                start: *s as usize,
                                end: *e as usize,
                                op: Arc::new(self.rebuild(*child)),
                            })
                            .collect(),
                    ),
                    InputRef::KeccakResult(v) => WrappedInput::KeccakResult(*v),
                })
                .collect();
            WrappedOpcode::new(*code, rebuilt_inputs)
        }
    }

    fn new_vm(bytecode: &[u8]) -> VM {
        VM::new(bytecode, &[], H160::zero(), H160::zero(), H160::zero(), 0, 1_000_000)
    }

    fn stream(bytecode: &[u8]) -> CollectingSink {
        let mut vm = new_vm(bytecode);
        let mut sink = CollectingSink::default();
        vm.stream_all_traces(
            None,
            None,
            None,
            false,
            None,
            &mut HashSet::new(),
            &mut sink,
        )
        .expect("stream_all_traces failed")
        .expect("expected a summary");
        sink
    }

    // PUSH1 0x00; CALLDATALOAD; PUSH1 0x08; JUMPI; STOP; pad; JUMPDEST; STOP
    // A dynamic JUMPI => two successors (continue + jump).
    const DYNAMIC_JUMPI: [u8; 10] = [0x60, 0x00, 0x35, 0x60, 0x08, 0x57, 0x00, 0x00, 0x5b, 0x00];

    #[test]
    fn determinism_across_runs() {
        let a = stream(&DYNAMIC_JUMPI);
        let b = stream(&DYNAMIC_JUMPI);

        let seg_ids_a: Vec<_> = a.segments.iter().map(|s| s.segment_hash).collect();
        let seg_ids_b: Vec<_> = b.segments.iter().map(|s| s.segment_hash).collect();
        assert_eq!(seg_ids_a, seg_ids_b, "segment ids differ across runs");

        let step_keys_a: Vec<_> =
            a.steps.iter().map(|s| (s.segment_hash, s.step_index, s.stack_id, s.memory_id)).collect();
        let step_keys_b: Vec<_> =
            b.steps.iter().map(|s| (s.segment_hash, s.step_index, s.stack_id, s.memory_id)).collect();
        assert_eq!(step_keys_a, step_keys_b, "step ids differ across runs");
    }

    #[test]
    fn interned_opcodes_rebuild_to_same_solidify() {
        let sink = stream(&DYNAMIC_JUMPI);
        // Every step's input/output operation ids must rebuild to a tree whose solidify() matches.
        let mut checked = 0usize;
        let mut vm = new_vm(&DYNAMIC_JUMPI);
        // Re-execute step by step and compare against the streamed records in order.
        for step in &sink.steps {
            let state = vm.step().expect("step failed");
            assert_eq!(state.last_instruction.instruction, step.instruction_pc);
            for (op, id) in
                state.last_instruction.input_operations.iter().zip(step.input_operation_ids.iter())
            {
                assert_eq!(op.solidify(), sink.rebuild(*id).solidify());
                checked += 1;
            }
            if state.last_instruction.opcode == 0x57 {
                break; // first segment ends at the JUMPI; that is enough to cover nested inputs.
            }
        }
        assert!(checked > 0, "expected to check at least one interned opcode");
    }

    #[test]
    fn chained_hash_continue_vs_jump() {
        let sink = stream(&DYNAMIC_JUMPI);
        let root = &sink.segments[0];
        assert_eq!(root.parent_hash, None);
        assert_eq!(root.children.len(), 2, "dynamic JUMPI must fork two children");

        let continue_child = root
            .children
            .iter()
            .find(|c| c.branch_kind == BranchKind::Continue)
            .expect("missing continue child");
        let jump_child = root
            .children
            .iter()
            .find(|c| c.branch_kind == BranchKind::Jump)
            .expect("missing jump child");

        // Continue hashes the parent id unchanged; jump hashes parent + 1.
        assert_eq!(continue_child.chained_hash, chained_segment_hash(root.segment_hash, 0));
        assert_eq!(jump_child.chained_hash, chained_segment_hash(root.segment_hash, 1));
        assert_ne!(continue_child.chained_hash, jump_child.chained_hash);
    }

    #[test]
    fn root_id_is_genesis_for_fresh_walk() {
        let sink = stream(&DYNAMIC_JUMPI);
        assert_eq!(sink.segments[0].segment_hash, genesis());
    }

    #[test]
    fn opcodes_emitted_children_before_parents() {
        let sink = stream(&DYNAMIC_JUMPI);
        // When an opcode is emitted, all of its Op-child ids must already have been emitted.
        let mut seen: HashSet<U256> = HashSet::new();
        for id in &sink.opcode_emit_order {
            let (_, inputs) = &sink.opcodes[id];
            for input in inputs {
                if let InputRef::Op(child) = input {
                    assert!(seen.contains(child), "parent opcode emitted before its child");
                }
            }
            seen.insert(*id);
        }
    }

    #[test]
    fn extend_route_reproduces_full_walk_id() {
        // A fresh full walk, then an extend along the route root -> (continue child). The route-end
        // node's chained id (folded along the route) must equal the id the full walk assigned.
        let full = stream(&DYNAMIC_JUMPI);
        let root = &full.segments[0];
        let continue_child = root
            .children
            .iter()
            .find(|c| c.branch_kind == BranchKind::Continue)
            .expect("missing continue child");
        let child_id = continue_child.chained_hash;
        let child_seg = full
            .segments
            .iter()
            .find(|s| s.segment_hash == child_id)
            .expect("continue child segment not emitted");

        // heimdall route convention: (first_pc, last_pc) are 0-indexed (== 1-indexed pointer - 1).
        let route = vec![
            (root.start_pc as usize - 1, root.end_pc as usize - 1),
            (child_seg.start_pc as usize - 1, child_seg.end_pc as usize - 1),
        ];

        let mut vm = new_vm(&DYNAMIC_JUMPI);
        let mut sink = CollectingSink::default();
        vm.stream_all_traces(
            None,
            None,
            None,
            false,
            Some(route),
            &mut HashSet::new(),
            &mut sink,
        )
        .expect("extend stream failed")
        .expect("expected a summary");

        // The extend's root is the route-end node and must carry the same id the full walk gave it.
        assert_eq!(sink.segments[0].segment_hash, child_id);
        assert_eq!(sink.segments[0].parent_hash, None);
    }

    #[test]
    fn loop_is_bounded_by_loop_limit() {
        // JUMPDEST; PUSH1 0x00; JUMP 0  — an unconditional self-loop. Without bounding this never
        // terminates; with loop_limit it unrolls a small, fixed number of times.
        let bytecode = [0x5b, 0x60, 0x00, 0x56];
        let mut vm = new_vm(&bytecode);
        let mut sink = CollectingSink::default();
        let summary = vm
            .stream_all_traces(
                None,
                None,
                Some(2),
                false,
                None,
                &mut HashSet::new(),
                &mut sink,
            )
            .expect("stream failed")
            .expect("expected a summary");
        assert!(summary.completed, "bounded loop should drain the queue");
        assert!(sink.segments.len() <= 8, "loop not bounded: {} segments", sink.segments.len());
    }

    #[test]
    fn shared_subexpr_with_distinct_raws_is_distinguished() {
        let mut sink = CollectingSink::default();
        let mut interner = Interner::default();

        // Shared child X = CALLDATALOAD(Raw(4)); the SAME Arc allocation is reused in both parents.
        let x = Arc::new(WrappedOpcode::new(0x35, vec![WrappedInput::Raw(U256::from(4))]));
        let e1 = WrappedOpcode::new(
            0x01, // ADD
            vec![WrappedInput::Opcode(x.clone()), WrappedInput::Raw(U256::from(1))],
        );
        let e2 = WrappedOpcode::new(
            0x01,
            vec![WrappedInput::Opcode(x.clone()), WrappedInput::Raw(U256::from(2))],
        );

        let id1 = interner.intern_opcode(&mut sink, &e1);
        let id2 = interner.intern_opcode(&mut sink, &e2);
        assert_ne!(id1, id2, "distinct raw operands must yield distinct ids");

        // The shared child is interned/emitted exactly once and both parents reference it.
        let x_id = interner.intern_opcode(&mut sink, &x);
        assert_eq!(
            sink.opcode_emit_order.iter().filter(|&&i| i == x_id).count(),
            1,
            "shared child must be emitted exactly once"
        );

        // Both parents rebuild to the original expression (and reuse the same child id).
        assert_eq!(sink.rebuild(id1).solidify(), e1.solidify());
        assert_eq!(sink.rebuild(id2).solidify(), e2.solidify());

        // A Raw literal can never alias an Op reference with the same 32-byte value (the input
        // tag byte separates them): ADD(Raw(x_id)) != ADD(Op(x)).
        let raw_lookalike = WrappedOpcode::new(0x01, vec![WrappedInput::Raw(x_id)]);
        let op_ref = WrappedOpcode::new(0x01, vec![WrappedInput::Opcode(x.clone())]);
        let id_raw = interner.intern_opcode(&mut sink, &raw_lookalike);
        let id_op = interner.intern_opcode(&mut sink, &op_ref);
        assert_ne!(id_raw, id_op, "Raw literal must not alias an Op reference with equal value");
    }

    #[test]
    fn static_false_jumpi_single_child() {
        // PUSH1 0x00 (cond=0); PUSH1 0x07 (dest); JUMPI; STOP; pad; JUMPDEST; STOP
        // never taken => only the fall-through (continue) successor.
        let bytecode = [0x60, 0x00, 0x60, 0x07, 0x57, 0x00, 0x00, 0x5b, 0x00];
        let sink = stream(&bytecode);
        let root = &sink.segments[0];
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].branch_kind, BranchKind::Continue);
        assert!(root.static_jumpi.is_some());
    }
}
