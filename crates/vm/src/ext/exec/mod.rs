mod jump_frame;
mod util;

use ethers::prelude::U256;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::{
    core::{
        opcodes::{Opcode, WrappedInput, WrappedOpcode},
        stack::Stack,
        vm::{State, VM},
    },
    ext::exec::{
        jump_frame::JumpFrame,
        util::{
            historical_diffs_approximately_equal, jump_condition_appears_recursive,
            jump_condition_contains_mutated_memory_access,
            jump_condition_contains_mutated_storage_access, stack_diff,
        },
    },
};
use eyre::{OptionExt, Result};
use heimdall_common::utils::strings::decode_hex;
use std::{collections::HashMap, mem::MaybeUninit, time::Instant};
use tracing::{debug, info, trace, warn};

#[derive(Clone, Debug, Default)]
pub struct VMTrace {
    pub instruction: u128,
    pub gas_used: u128,
    pub operations: Vec<State>,
    pub children: Vec<VMTrace>,
}

#[derive(Clone, Debug, Default)]
pub struct VMTraceExtended {
    pub id: u32,
    pub hash: U256,
    pub children: Vec<VMTraceExtended>,
    pub next_possible_segment_hashes: HashSet<U256>,
}

#[derive(Clone, Debug, Default)]
struct VmSnapshotProfile {
    stack_frames: usize,
    stack_max_op_depth: u32,
    stack_total_op_nodes: usize,
    memory_bytes: usize,
    memory_op_entries: usize,
    memory_max_op_depth: u32,
    memory_total_op_nodes: usize,
    storage_slots: usize,
    transient_slots: usize,
    event_count: usize,
}

#[derive(Clone, Debug, Default)]
struct SegmentTraceProfile {
    // how many operations are in this trace / segment
    op_count: usize,
    // the totak length for each state in this trace / segment
    total_state_memory_bytes: usize,
    total_state_stack_frames: usize,
    total_state_storage_slots: usize,
    total_state_transient_slots: usize,
    total_state_event_count: usize,
    //
    max_state_memory_bytes: usize,
    max_state_stack_frames: usize,
    max_state_storage_slots: usize,
    max_state_transient_slots: usize,
    // the maximum depth of input WrappedOpcode in this trace / segment
    max_state_input_op_depth: u32,
    // output
    max_state_output_op_depth: u32,
    max_state_stack_op_depth: u32,
}

#[derive(Clone, Debug, Default)]
struct PathTraceProfile {
    segment_count: usize,
    total_ops: usize,
    total_state_memory_bytes: usize,
    total_state_stack_frames: usize,
    total_state_storage_slots: usize,
    total_state_transient_slots: usize,
    total_state_event_count: usize,
    max_segment_ops: usize,
    max_segment_total_state_memory_bytes: usize,
    max_segment_max_state_memory_bytes: usize,
    max_segment_max_state_stack_frames: usize,
    max_segment_max_state_storage_slots: usize,
    max_segment_max_state_transient_slots: usize,
    max_input_op_depth: u32,
    max_output_op_depth: u32,
    max_stack_op_depth: u32,
}

#[derive(Clone, Debug, Default)]
struct TrimTraceProfile {
    original_op_count: usize,
    retained_op_count: usize,
    removed_op_count: usize,
    cleared_stack_frames: usize,
}

fn wrapped_opcode_max_depth(opcode: &WrappedOpcode) -> u32 {
    opcode.depth()
}

fn wrapped_opcode_slice_max_depth(opcodes: &[WrappedOpcode]) -> u32 {
    opcodes.iter().map(wrapped_opcode_max_depth).max().unwrap_or(0)
}

fn stack_max_op_depth(stack: &Stack) -> u32 {
    stack.stack.iter().map(|frame| frame.operation.depth()).max().unwrap_or(0)
}

fn wrapped_opcode_node_count(opcode: &WrappedOpcode) -> usize {
    1 + opcode.inputs.iter().map(|input| match input {
        WrappedInput::Opcode(op) => wrapped_opcode_node_count(op),
        WrappedInput::Raw(_) => 0,
    }).sum::<usize>()
}

fn stack_total_op_nodes(stack: &Stack) -> usize {
    stack.stack.iter().map(|frame| wrapped_opcode_node_count(&frame.operation)).sum()
}

#[cfg(feature = "experimental")]
fn memory_max_op_depth(vm: &VM) -> u32 {
    vm.memory.bytes.0.values().map(|op| op.depth()).max().unwrap_or(0)
}

#[cfg(feature = "experimental")]
fn memory_total_op_nodes(vm: &VM) -> usize {
    vm.memory.bytes.0.values().map(|op| wrapped_opcode_node_count(op)).sum()
}

fn vm_snapshot_profile(vm: &VM) -> VmSnapshotProfile {
    VmSnapshotProfile {
        stack_frames: vm.stack.stack.len(),
        stack_max_op_depth: stack_max_op_depth(&vm.stack),
        stack_total_op_nodes: stack_total_op_nodes(&vm.stack),
        memory_bytes: vm.memory.memory.len(),
        #[cfg(feature = "experimental")]
        memory_op_entries: vm.memory.bytes.0.len(),
        #[cfg(not(feature = "experimental"))]
        memory_op_entries: 0,
        #[cfg(feature = "experimental")]
        memory_max_op_depth: memory_max_op_depth(vm),
        #[cfg(not(feature = "experimental"))]
        memory_max_op_depth: 0,
        #[cfg(feature = "experimental")]
        memory_total_op_nodes: memory_total_op_nodes(vm),
        #[cfg(not(feature = "experimental"))]
        memory_total_op_nodes: 0,
        storage_slots: vm.storage.storage.len(),
        transient_slots: vm.storage.transient.len(),
        event_count: vm.events.len(),
    }
}

fn process_peak_rss_kb() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return None;
    }

    let usage = unsafe { usage.assume_init() };
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some((usage.ru_maxrss as u64).saturating_add(1023) / 1024)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Some(usage.ru_maxrss as u64)
    }
}

fn segment_trace_profile(trace: &VMTrace) -> SegmentTraceProfile {
    let mut profile =
        SegmentTraceProfile { op_count: trace.operations.len(), ..Default::default() };

    for state in &trace.operations {
        let memory_bytes = state.memory.memory.len();
        let stack_frames = state.stack.stack.len();
        let storage_slots = state.storage.storage.len();
        let transient_slots = state.storage.transient.len();
        let event_count = state.events.len();

        profile.total_state_memory_bytes += memory_bytes;
        profile.total_state_stack_frames += stack_frames;
        profile.total_state_storage_slots += storage_slots;
        profile.total_state_transient_slots += transient_slots;
        profile.total_state_event_count += event_count;

        profile.max_state_memory_bytes = profile.max_state_memory_bytes.max(memory_bytes);
        profile.max_state_stack_frames = profile.max_state_stack_frames.max(stack_frames);
        profile.max_state_storage_slots = profile.max_state_storage_slots.max(storage_slots);
        profile.max_state_transient_slots = profile.max_state_transient_slots.max(transient_slots);

        profile.max_state_input_op_depth = profile
            .max_state_input_op_depth
            .max(wrapped_opcode_slice_max_depth(&state.last_instruction.input_operations));
        profile.max_state_output_op_depth = profile
            .max_state_output_op_depth
            .max(wrapped_opcode_slice_max_depth(&state.last_instruction.output_operations));
        profile.max_state_stack_op_depth =
            profile.max_state_stack_op_depth.max(stack_max_op_depth(&state.stack));
    }

    profile
}

fn accumulate_path_trace_profile(
    aggregate: &mut PathTraceProfile,
    segment_profile: &SegmentTraceProfile,
) {
    aggregate.segment_count += 1;
    aggregate.total_ops += segment_profile.op_count;
    aggregate.total_state_memory_bytes += segment_profile.total_state_memory_bytes;
    aggregate.total_state_stack_frames += segment_profile.total_state_stack_frames;
    aggregate.total_state_storage_slots += segment_profile.total_state_storage_slots;
    aggregate.total_state_transient_slots += segment_profile.total_state_transient_slots;
    aggregate.total_state_event_count += segment_profile.total_state_event_count;
    aggregate.max_segment_ops = aggregate.max_segment_ops.max(segment_profile.op_count);
    aggregate.max_segment_total_state_memory_bytes = aggregate
        .max_segment_total_state_memory_bytes
        .max(segment_profile.total_state_memory_bytes);
    aggregate.max_segment_max_state_memory_bytes =
        aggregate.max_segment_max_state_memory_bytes.max(segment_profile.max_state_memory_bytes);
    aggregate.max_segment_max_state_stack_frames =
        aggregate.max_segment_max_state_stack_frames.max(segment_profile.max_state_stack_frames);
    aggregate.max_segment_max_state_storage_slots =
        aggregate.max_segment_max_state_storage_slots.max(segment_profile.max_state_storage_slots);
    aggregate.max_segment_max_state_transient_slots = aggregate
        .max_segment_max_state_transient_slots
        .max(segment_profile.max_state_transient_slots);
    aggregate.max_input_op_depth =
        aggregate.max_input_op_depth.max(segment_profile.max_state_input_op_depth);
    aggregate.max_output_op_depth =
        aggregate.max_output_op_depth.max(segment_profile.max_state_output_op_depth);
    aggregate.max_stack_op_depth =
        aggregate.max_stack_op_depth.max(segment_profile.max_state_stack_op_depth);
}

fn is_key_opcode(opcode: u8) -> bool {
    matches!(
        opcode,
        0x55 // SSTORE
            | 0x54 // SLOAD
            | 0x56 // JUMP
            | 0x57 // JUMPI
            | 0xf1 // CALL
            | 0xfa // STATICCALL
            | 0x00 // STOP
            | 0xf3 // RETURN
            | 0xfd // REVERT
            | 0xfe // INVALID
            | 0xff // SELFDESTRUCT
            | 0x5b // JUMPDEST
            | 0x10 // LT
            | 0x11 // GT
            | 0x12 // SLT
            | 0x13 // SGT
            | 0x14 // EQ
            | 0x15 // ISZERO
            | 0xa1 // LOG1
            | 0x52 // MSTORE
            | 0x53 // MSTORE8
            | 0x51 // MLOAD
            | 0x3d // RETURNDATASIZE
            | 0x3e // RETURNDATACOPY
            | 0x20 // KECCAK256
            | 0x35 // CALLDATALOAD
            | 0x37 // CALLDATACOPY
    )
}

fn trim_trace_for_storage(trace: &mut VMTrace) -> TrimTraceProfile {
    let original_op_count = trace.operations.len();
    if original_op_count == 0 {
        return TrimTraceProfile::default();
    }

    let last_idx = original_op_count - 1;
    let mut cleared_stack_frames = 0usize;
    trace.operations = trace
        .operations
        .drain(..)
        .enumerate()
        .filter_map(|(idx, mut state)| {
            let should_keep = idx == 0
                || idx == last_idx
                || is_key_opcode(state.last_instruction.opcode);
            if !should_keep {
                return None;
            }

            for frame in state.stack.stack.iter_mut() {
                frame.operation = WrappedOpcode::default();
                cleared_stack_frames += 1;
            }
            Some(state)
        })
        .collect();

    let retained_op_count = trace.operations.len();
    TrimTraceProfile {
        original_op_count,
        retained_op_count,
        removed_op_count: original_op_count.saturating_sub(retained_op_count),
        cleared_stack_frames,
    }
}

impl VM {
    /// Run symbolic execution on a given function selector within a contract
    pub fn symbolic_exec_selector(
        &mut self,
        selector: &str,
        entry_point: u128,
        timeout: Instant,
    ) -> Result<(VMTrace, u32)> {
        self.calldata = decode_hex(selector)?;

        // step through the bytecode until we reach the entry point
        while self.bytecode.len() >= self.instruction as usize && (self.instruction <= entry_point)
        {
            self.step()?;

            // this shouldn't be necessary, but it's safer to have it
            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        trace!("beginning symbolic execution for selector 0x{}", selector);

        // the VM is at the function entry point, begin tracing
        let mut branch_count = 0;
        Ok((
            self.recursive_map(&mut branch_count, &mut HashMap::new(), &timeout)
                .map(|x| x.ok_or_eyre("symbolic execution failed"))??,
            branch_count,
        ))
    }

    // build a map of function jump possibilities from the EVM bytecode
    pub fn symbolic_exec(&mut self, timeout: Instant) -> Result<(VMTrace, u32)> {
        trace!("beginning contract-wide symbolic execution");

        // the VM is at the function entry point, begin tracing
        let mut branch_count = 0;
        Ok((
            self.recursive_map(&mut branch_count, &mut HashMap::new(), &timeout)
                .map(|x| x.ok_or_eyre("symbolic execution failed"))??,
            branch_count,
        ))
    }

    fn recursive_map(
        &mut self,
        branch_count: &mut u32,
        handled_jumps: &mut HashMap<JumpFrame, Vec<Stack>>,
        timeout_at: &Instant,
    ) -> Result<Option<VMTrace>> {
        let vm = self;

        // create a new VMTrace object
        // this will essentially be a tree of executions, with each branch being a different path
        // that symbolic execution discovered
        let mut vm_trace = VMTrace {
            instruction: vm.instruction,
            gas_used: 0,
            operations: Vec::new(),
            children: Vec::new(),
        };

        // step through the bytecode until we find a JUMPI instruction
        while vm.bytecode.len() >= vm.instruction as usize {
            // if we have reached the timeout, return None
            if Instant::now() >= *timeout_at {
                return Ok(Some(vm_trace));
            }

            // execute the next instruction. if the instruction panics, invalidate this path
            let state = vm.step()?;
            let last_instruction = state.last_instruction.clone();

            // update vm_trace
            vm_trace.operations.push(state);
            vm_trace.gas_used = vm.gas_used;

            // if we encounter a JUMP(I), create children taking both paths and break
            if last_instruction.opcode == 0x57 {
                trace!(
                    "found branch due to JUMP{} instruction at {}",
                    if last_instruction.opcode == 0x57 { "I" } else { "" },
                    last_instruction.instruction
                );

                let jump_condition: Option<String> =
                    last_instruction.input_operations.get(1).map(|op| op.solidify());
                let jump_taken =
                    last_instruction.inputs.get(1).map(|op| !op.is_zero()).unwrap_or(true);

                // build hashable jump frame
                let jump_frame = JumpFrame::new(
                    last_instruction.instruction,
                    last_instruction.inputs[0],
                    vm.stack.size(),
                    jump_taken,
                );

                // if the stack contains too many items, it's probably a loop
                // if stack_contains_too_many_items(&vm.stack) {
                //     return Ok(Some(vm_trace));
                // }

                // if the stack has over 16 items of the same source, it's probably a loop
                // if stack_contains_too_many_of_the_same_item(&vm.stack) {
                //     return Ok(Some(vm_trace));
                // }

                // if any item on the stack has a depth > 16, it's probably a loop (because of stack
                // too deep)
                // if stack_item_source_depth_too_deep(&vm.stack) {
                //     return Ok(Some(vm_trace));
                // }

                // if the jump stack depth is less than the max stack depth of all previous matching
                // jumps, it's probably a loop
                // if jump_stack_depth_less_than_max_stack_depth(&jump_frame, handled_jumps) {
                //     return Ok(Some(vm_trace));
                // }

                // perform heuristic checks on historical stacks
                match handled_jumps.get_mut(&jump_frame) {
                    Some(historical_stacks) => {
                        // for every stack that we have encountered for this jump, perform some
                        // heuristic checks to determine if this might be a loop
                        if historical_stacks.iter().any(|hist_stack| {
                            if let Some(jump_condition) = &jump_condition {

                                // check if any historical stack is the same as the current stack
                                if hist_stack == &vm.stack {
                                    trace!(
                                        "jump matches loop-detection heuristic: 'jump_path_already_handled'"
                                    );
                                    return true
                                }

                                // calculate the difference of the current stack and the historical stack
                                let stack_diff = stack_diff(&vm.stack, hist_stack);
                                if stack_diff.is_empty() {
                                    // the stack_diff is empty (the stacks are the same), so we've
                                    // already handled this path
                                    trace!(
                                        "jump matches loop-detection heuristic: 'stack_diff_is_empty'"
                                    );
                                    return true
                                }

                                trace!("stack diff: [{}]", stack_diff.iter().map(|frame| format!("{}", frame.value)).collect::<Vec<String>>().join(", "));

                                // check if the jump condition appears to be recursive
                                if jump_condition_appears_recursive(&stack_diff, jump_condition) {
                                    return true
                                }

                                // check for mutated memory accesses in the jump condition
                                if jump_condition_contains_mutated_memory_access(
                                    &stack_diff,
                                    jump_condition,
                                ) {
                                    return true
                                }

                                // check for mutated memory accesses in the jump condition
                                if jump_condition_contains_mutated_storage_access(
                                    &stack_diff,
                                    jump_condition,
                                ) {
                                    return true
                                }

                            }
                            false
                        }) {
                            trace!("jump terminated.");
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                            return Ok(Some(vm_trace));
                        }

                        if historical_diffs_approximately_equal(&vm.stack, historical_stacks) {
                            trace!("jump terminated.");
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                            return Ok(Some(vm_trace));
                        } else {
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );
                            trace!(
                                " - jump condition: {:?}\n        - stack: {}\n        - historical stacks: {}",
                                jump_condition,
                                vm.stack,
                                historical_stacks.iter().map(|stack| format!("{}", stack)).collect::<Vec<String>>().join("\n            - ")
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                        }
                    }
                    None => {
                        // this key doesnt exist, so the jump is new
                        trace!("added new jump frame: {:?}", jump_frame);
                        handled_jumps.insert(jump_frame, vec![vm.stack.clone()]);
                    }
                }

                if last_instruction.opcode == 0x56 {
                    continue;
                }

                // we didnt break out, so now we crate branching paths to cover all possibilities
                *branch_count += 1;
                trace!(
                    "creating branching paths at instructions {} (JUMPDEST) and {} (CONTINUE)",
                    last_instruction.inputs[0],
                    last_instruction.instruction + 1
                );

                // we need to create a trace for the path that wasn't taken.
                if !jump_taken {
                    // push a new vm trace to the children
                    let mut trace_vm = vm.clone();
                    trace_vm.instruction = last_instruction.inputs[0].as_u128() + 1;
                    match trace_vm.recursive_map(branch_count, handled_jumps, timeout_at) {
                        Ok(Some(child_trace)) => vm_trace.children.push(child_trace),
                        Ok(None) => {}
                        Err(e) => {
                            warn!("error executing branch: {:?}", e);
                            return Ok(Some(vm_trace));
                        }
                    }

                    // push the current path onto the stack
                    match vm.recursive_map(branch_count, handled_jumps, timeout_at) {
                        Ok(Some(child_trace)) => vm_trace.children.push(child_trace),
                        Ok(None) => {}
                        Err(e) => {
                            warn!("error executing branch: {:?}", e);
                            return Ok(Some(vm_trace));
                        }
                    }
                    break;
                } else {
                    // push a new vm trace to the children
                    let mut trace_vm = vm.clone();
                    trace_vm.instruction = last_instruction.instruction + 1;
                    match trace_vm.recursive_map(branch_count, handled_jumps, timeout_at) {
                        Ok(Some(child_trace)) => vm_trace.children.push(child_trace),
                        Ok(None) => {}
                        Err(e) => {
                            warn!("error executing branch: {:?}", e);
                            return Ok(Some(vm_trace));
                        }
                    }

                    // push the current path onto the stack
                    match vm.recursive_map(branch_count, handled_jumps, timeout_at) {
                        Ok(Some(child_trace)) => vm_trace.children.push(child_trace),
                        Ok(None) => {}
                        Err(e) => {
                            warn!("error executing branch: {:?}", e);
                            return Ok(Some(vm_trace));
                        }
                    }
                    break;
                }
            }

            // when the vm exits, this path is complete
            if vm.exitcode != 255 || !vm.returndata.is_empty() {
                break;
            }
        }

        Ok(Some(vm_trace))
    }

    fn jump_stack_hash_helper(
        jump_dest_pc: &HashSet<U256>,
        pc: u128,
        stack: &Stack,
    ) -> U256 {
        let t_solidify = Instant::now();
        let mut hash_data: Vec<U256> = Vec::new();
        hash_data.push(U256::from(pc));
        for (i, s) in stack.stack.iter().enumerate() {
            let solidified_operation = s.operation.solidify();
            if jump_dest_pc.contains(&s.value) && solidified_operation.starts_with("0x") && !solidified_operation.contains(" ") {
                hash_data.push(U256::from(i));
                hash_data.push(s.value);
            }
        }
        let solidify_ms = t_solidify.elapsed().as_secs_f64() * 1000.0;

        let t_keccak = Instant::now();
        let mut data: Vec<u8> = Vec::new();
        for v in &hash_data {
            data.append(&mut v.to_string().into_bytes());
        }
        let jump_and_jumpi_hash = U256::from(ethers::core::utils::keccak256(&data));
        let keccak_ms = t_keccak.elapsed().as_secs_f64() * 1000.0;

        if solidify_ms + keccak_ms > 2000.0 {
            debug!(
                "[heimdall] jump_stack_hash_helper: pc={} stack_frames={} \
                 solidify_ms={:.2} keccak_ms={:.2}",
                pc, stack.stack.len(), solidify_ms, keccak_ms,
            );
        }

        jump_and_jumpi_hash
    }

    // generate a safe node id by route
    fn generate_safe_node_id_by_route(
        route: Vec<(usize, usize)>,
    ) -> u32 {
        let mut safe_node_id = 0;
        let mut data: Vec<u8> = Vec::new();
        for node in route.clone() {
            data.append(&mut node.0.to_be_bytes().to_vec());
            data.append(&mut node.1.to_be_bytes().to_vec());
        }

        loop {
            let hash = U256::from(ethers::core::utils::keccak256(&data));
            let hash_within_u32 = hash % U256::from(u32::MAX);
            safe_node_id = hash_within_u32.as_u32();
            if u32::MAX - safe_node_id > 1_000_000 {
                break;
            } else {
                data.append(&mut safe_node_id.to_be_bytes().to_vec());
            }
        }

        safe_node_id
    }

    fn program_counter(contract_bytecode: Vec<u8>) -> HashMap<U256, Opcode> {
        let mut program_counter = 0;
        let mut pc_n_opcode: HashMap<U256, Opcode> = HashMap::new();
        while program_counter < contract_bytecode.len() {
            let operation = Opcode::new(contract_bytecode[program_counter]);
            let current_pc = program_counter;

            // handle PUSH0 -> PUSH32, which require us to push the next N bytes
            // onto the stack
            if operation.code >= 0x5f && operation.code <= 0x7f {
                let byte_count_to_push: u8 = operation.code - 0x5f;
                program_counter += byte_count_to_push as usize;
            }
            pc_n_opcode.insert(U256::from(current_pc), operation);

            program_counter += 1;
        }

        pc_n_opcode
    }

    // build a trace from current instruction and return the next traces to explore
    fn build_trace(&mut self) -> Result<(VMTrace, Vec<VM>)> {
        // let build_trace_start = Instant::now();
        // let instruction_before = self.instruction;
        // let vm_profile_before = vm_snapshot_profile(self);
        
        let mut root_trace = VMTrace {
            instruction: self.instruction,
            gas_used: 0,
            operations: Vec::new(),
            children: Vec::new(),
        };

        let mut next_traces = Vec::new();
        while self.bytecode.len() >= self.instruction as usize {
            let state = self.step()?;
            let last_instruction = state.last_instruction.clone();
            root_trace.operations.push(state);
            root_trace.gas_used = self.gas_used;

            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }

            // jump / jumpi
            if last_instruction.opcode == 0x57 || last_instruction.opcode == 0x56 {
                if last_instruction.opcode == 0x57 {
                    // continue branch
                    let mut new_trace = self.clone();
                    new_trace.instruction = last_instruction.instruction + 1;
                    next_traces.push(new_trace);
                }

                // jump branch
                let mut new_trace = self.clone();
                new_trace.instruction = last_instruction.inputs[0].as_u128() + 1;
                next_traces.push(new_trace);
                break;
            }

            // next instruction is jumpdest
            if self
                .bytecode
                .get((self.instruction - 1) as usize)
                .ok_or_eyre(format!("invalid jumpdest: {}", self.instruction - 1))?
            .to_owned() == 0x5b {
                next_traces.push(self.clone());
                break;
            }
        }

        // let segment_profile = segment_trace_profile(&root_trace);
        // let vm_profile_after = vm_snapshot_profile(self);
        // let build_trace_duration = build_trace_start.elapsed();
        // let op_count = segment_profile.op_count.max(1);
        // let last_pc = root_trace
        //     .operations
        //     .last()
        //     .map(|state| state.last_instruction.instruction.saturating_sub(1))
        //     .unwrap_or_else(|| instruction_before.saturating_sub(1));
        // debug!(
        //     "[heimdall] build_trace summary: start_pc={} end_pc={} ops={} next_traces={} seg_state_mem_bytes={} seg_state_stack_frames={} seg_state_storage_slots={} seg_state_transient_slots={} seg_max_state_mem_bytes={} seg_max_state_stack_frames={} seg_max_state_storage_slots={} seg_max_state_transient_slots={} seg_max_input_op_depth={} seg_max_output_op_depth={} seg_max_stack_op_depth={} vm_stack_frames_before={} vm_stack_frames_after={} vm_stack_max_op_depth_before={} vm_stack_max_op_depth_after={} vm_memory_bytes_before={} vm_memory_bytes_after={} vm_storage_slots_before={} vm_storage_slots_after={} vm_transient_slots_before={} vm_transient_slots_after={} vm_event_count_before={} vm_event_count_after={} peak_rss_kb={:?} duration_ms={} ms_per_op={:.3}",
        //     instruction_before.saturating_sub(1),
        //     last_pc,
        //     segment_profile.op_count,
        //     next_traces.len(),
        //     segment_profile.total_state_memory_bytes,
        //     segment_profile.total_state_stack_frames,
        //     segment_profile.total_state_storage_slots,
        //     segment_profile.total_state_transient_slots,
        //     segment_profile.max_state_memory_bytes,
        //     segment_profile.max_state_stack_frames,
        //     segment_profile.max_state_storage_slots,
        //     segment_profile.max_state_transient_slots,
        //     segment_profile.max_state_input_op_depth,
        //     segment_profile.max_state_output_op_depth,
        //     segment_profile.max_state_stack_op_depth,
        //     vm_profile_before.stack_frames,
        //     vm_profile_after.stack_frames,
        //     vm_profile_before.stack_max_op_depth,
        //     vm_profile_after.stack_max_op_depth,
        //     vm_profile_before.memory_bytes,
        //     vm_profile_after.memory_bytes,
        //     vm_profile_before.storage_slots,
        //     vm_profile_after.storage_slots,
        //     vm_profile_before.transient_slots,
        //     vm_profile_after.transient_slots,
        //     vm_profile_before.event_count,
        //     vm_profile_after.event_count,
        //     process_peak_rss_kb(),
        //     build_trace_duration.as_millis(),
        //     build_trace_duration.as_secs_f64() * 1000.0 / op_count as f64,
        // );

        Ok((root_trace, next_traces))
    }

    // build all traces from the current instruction according to the branch, segment, and loop limits
    // - The branch limit is the maximum number of branches to explore. If not provided, it will be ignored.
    // - The segment limit is the maximum number of segments to explore. If not provided, it will be ignored.
    // - The loop limit is the maximum number of times a segment can appear in the branch. If not provided, it will be set to 1.
    // - If simple_cfg is true, we will only process each similar trace once by checking globally and ignore the branch and segment limits.
    // - When return None, it means we have reached the branch or segment limits.
    pub fn build_all_traces(&mut self,
        branch_limit: Option<u32>,
        segment_limit: Option<u32>,
        loop_limit: Option<u32>,
        simple_cfg: bool,
        route: Option<Vec<(usize, usize)>>,
        processed_nodes: &mut HashSet<U256>,
    ) -> Result<Option<(VMTrace, VMTraceExtended)>> {
        let build_all_traces_start = Instant::now();
        let route_len = route.as_ref().map(|route| route.len()).unwrap_or(0);
        let mut branch_count: u32 = 0;
        let mut segment_count: u32 = 0;

        let jumpdest_pc = Self::program_counter(self.bytecode.clone())
            .iter()
            .filter(|(k, v)| v.code == 0x5b)
            .map(|(k, _)| k.clone())
            .collect::<HashSet<U256>>();

        let mut node_counter: u32 = 0;
        // this hash means the stack before the instruction is executed
        let root_trace_hash = Self::jump_stack_hash_helper(&jumpdest_pc,
            self.instruction, 
            &self.stack);
        let initial_route_or_root_trace_start = Instant::now();
        let (root_trace, mut next_traces, route_hashes) = if let Some(route) = route {
            node_counter = Self::generate_safe_node_id_by_route(route.clone());
            self.build_trace_start_from_route(route, &jumpdest_pc)?
        } else {
            let (t, v) = self.build_trace()?;
            (t, v, HashMap::new())
        };
        debug!(
            "[heimdall] initial_root_trace: route_len={} simple_cfg={} initial_next_traces_len={} duration_ms={}",
            route_len,
            simple_cfg,
            next_traces.len(),
            initial_route_or_root_trace_start.elapsed().as_millis()
        );
        {
            let rss_kb = process_peak_rss_kb().unwrap_or(0);
            debug!(
                "[heimdall] build_all_traces post_root_trace: rss_kb={} queue_init={}",
                rss_kb,
                next_traces.len(),
            );
        }
        let root_node_id = node_counter;

        let next_possible_segment_hashes_fn = |trace: &VMTrace| -> Result<HashSet<U256>> {
            let mut hashes = HashSet::new();
            let opcode = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.opcode as u128;
            let last_instruction = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.instruction;
            match opcode {
                0x57_u128 => {
                    hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc,
                        trace.operations.last().ok_or_eyre("no operations")?.last_instruction.inputs[0].as_u128() + 1, 
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                    hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc,
                        last_instruction + 1,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
                0x56_u128 => {
                    hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc,
                        trace.operations.last().ok_or_eyre("no operations")?.last_instruction.inputs[0].as_u128() + 1, 
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
                0x5f_u128 | 0x60_u128 | 0x61_u128 | 0x62_u128 | 0x63_u128 | 0x64_u128 | 0x65_u128 |
                0x66_u128 | 0x67_u128 | 0x68_u128 | 0x69_u128 | 0x6a_u128 | 0x6b_u128 | 0x6c_u128 |
                0x6d_u128 | 0x6e_u128 | 0x6f_u128 | 0x70_u128 | 0x71_u128 | 0x72_u128 | 0x73_u128 |
                0x74_u128 | 0x75_u128 | 0x76_u128 | 0x77_u128 | 0x78_u128 | 0x79_u128 | 0x7a_u128 |
                0x7b_u128 | 0x7c_u128 | 0x7d_u128 | 0x7e_u128 | 0x7f_u128 => {
                    let next_instruction = last_instruction + (opcode as u128 - 0x5f) + 1;
                    let hash = Self::jump_stack_hash_helper(&jumpdest_pc,
                        next_instruction,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack);                    
                    hashes.insert(hash);
                }
                _ => {
                    hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc,
                        last_instruction + 1,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
            }
            Ok(hashes)
        };
        let root_next_possible_segment_hashes = next_possible_segment_hashes_fn(&root_trace).map_err(|e| eyre::eyre!("failed to get next possible segment hashes: {}", e))?;
        let mut root_trace = root_trace;
        let mut trim_profile = trim_trace_for_storage(&mut root_trace);

        // pre-seed route segment hashes so loop detection does not miss them during re-exploration
        let mut previous_trace_hash = route_hashes;
        *previous_trace_hash.entry(root_trace_hash.clone()).or_insert(0) += 1;

        // update the branch and segment counts for the root trace
        branch_count += 1;
        segment_count += 1;

        let mut parent_to_children: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut node_entries_by_id: HashMap<u32, (Option<u32>, VMTraceExtended, VMTrace)> = HashMap::new(); // (parent_id, trace_hash, trace)
        node_entries_by_id.insert(node_counter, (None,
                VMTraceExtended {
                    id: node_counter,
                    hash: root_trace_hash,
                    children: Vec::new(),
                    next_possible_segment_hashes: root_next_possible_segment_hashes,
                },
                root_trace,
        ));
        parent_to_children.entry(node_counter).or_insert(HashSet::new());

        // initialize the queue with the first set of traces
        let mut queue: VecDeque<(u32, HashMap<U256, u32>, VM)> = VecDeque::new();
        for next_trace in next_traces.drain(..) {
            queue.push_back((node_counter, previous_trace_hash.clone(), next_trace));
        }

        // process the queue until it is empty
        let queue_expand_start = Instant::now();
        let mut queue_iterations = 0usize;
        while !queue.is_empty() {
            queue_iterations += 1;
            if queue_iterations % 500 == 0 {
                let rss_kb = process_peak_rss_kb().unwrap_or(0);
                debug!(
                    "[heimdall] build_all_traces queue: route_len={} simple_cfg={} queue_iter={} queue_size={} node_entries={} segment_count={} branch_count={} processed_nodes={} rss_kb={} elapsed_ms={}",
                    route_len, simple_cfg, queue_iterations, queue.len(),
                    node_entries_by_id.len(),
                    segment_count, branch_count, processed_nodes.len(),
                    rss_kb,
                    queue_expand_start.elapsed().as_millis()
                );
            }
            if !simple_cfg {
                if branch_limit.is_some() && branch_count >= branch_limit.unwrap() {
                    return Ok(None);
                }
                if segment_limit.is_some() && segment_count >= segment_limit.unwrap() {
                    return Ok(None);
                }
            } else {
                if branch_limit.is_some() && branch_count >= branch_limit.unwrap() {
                    warn!(
                        "[heimdall] build_all_traces stop by branch limit: route_len={} queue_iterations={} branch_count={} limit={} breaking early to avoid OOM",
                        route_len, queue_iterations, branch_count, branch_limit.unwrap()
                    );
                    break;
                }
                if segment_limit.is_some() && segment_count >= segment_limit.unwrap() {
                    warn!(
                        "[heimdall] build_all_traces stop by segment limit: route_len={} queue_iterations={} segment_count={} limit={} breaking early to avoid OOM",
                        route_len, queue_iterations, segment_count, segment_limit.unwrap()
                    );
                    break;
                }
            }

            let (parent_id, mut previous_trace_hash, mut vm) = queue.pop_front().ok_or_eyre("no next traces")?;
            // this hash means the stack before the instruction is executed
            let current_trace_hash = Self::jump_stack_hash_helper(&jumpdest_pc,
                vm.instruction, 
                &vm.stack);
            let (mut trace, mut next_traces) = vm.build_trace()?;

            // loop detection
            let updated_count = {
                let count = previous_trace_hash.entry(current_trace_hash).or_insert(0);
                *count += 1;
                *count
            };
            let loop_limit = loop_limit.unwrap_or(1);
            // validate with loop detection heuristics. if the trace is a loop, skip it
            if updated_count > loop_limit {
                continue;
            }

            // if we are building a simple cfg, we only want to process each similar trace once by checking globally
            if simple_cfg {
                // loop segments (updated count > 1) that passed loop detection above
                // should not be additionally blocked
                let is_known_loop_segment = updated_count > 1;
                if !processed_nodes.contains(&current_trace_hash) {
                    processed_nodes.insert(current_trace_hash);
                } else if parent_id == root_node_id || is_known_loop_segment {
                    // allow: direct child of root, or known loop segment within loop_limit
                } else {
                    continue;
                }
            }

            if next_traces.len() > 1 {
                branch_count += 1;
            }
            segment_count += 1;
            node_counter += 1;
            let next_possible_segment_hashes = next_possible_segment_hashes_fn(&trace).map_err(|e| eyre::eyre!("failed to get next possible segment hashes: {}", e))?;
            let segment_trim_profile = trim_trace_for_storage(&mut trace);
            trim_profile.original_op_count += segment_trim_profile.original_op_count;
            trim_profile.retained_op_count += segment_trim_profile.retained_op_count;
            trim_profile.removed_op_count += segment_trim_profile.removed_op_count;
            trim_profile.cleared_stack_frames += segment_trim_profile.cleared_stack_frames;
            node_entries_by_id.insert(node_counter,
                (Some(parent_id),
                    VMTraceExtended {
                        id: node_counter,
                        hash: current_trace_hash,
                        children: Vec::new(),
                        next_possible_segment_hashes,
                    },
                    trace,
                ),
            );
            parent_to_children.entry(parent_id).or_insert(HashSet::new()).insert(node_counter);
            parent_to_children.entry(node_counter).or_insert(HashSet::new());
            for next_trace in next_traces.drain(..) {
                queue.push_back((node_counter, previous_trace_hash.clone(), next_trace));
            }
        }
        {
            let rss_kb = process_peak_rss_kb().unwrap_or(0);
            debug!(
                "[heimdall] build_all_traces post_queue: route_len={} simple_cfg={} queue_iterations={} segment_count={} branch_count={} node_entries={} processed_nodes={} trimmed_original_ops={} trimmed_retained_ops={} trimmed_removed_ops={} trimmed_cleared_stack_frames={} rss_kb={} duration_ms={}",
                route_len,
                simple_cfg,
                queue_iterations,
                segment_count,
                branch_count,
                node_entries_by_id.len(),
                processed_nodes.len(),
                trim_profile.original_op_count,
                trim_profile.retained_op_count,
                trim_profile.removed_op_count,
                trim_profile.cleared_stack_frames,
                rss_kb,
                queue_expand_start.elapsed().as_millis()
            );
        }

        // always start with the nodes that have no children
        let mut ids = parent_to_children.iter().filter_map(|(parent_id, children)| {
                    if children.is_empty() {
                        Some(*parent_id)
                    } else {
                        None
                    }
        }).collect::<Vec<u32>>();

        // sort the ids to ensure we process the nodes in a consistent order
        ids.sort();

        let mut root_trace: Option<VMTrace> = None;
        let mut root_vm_trace_hash: Option<VMTraceExtended> = None;
        while !ids.is_empty() {
            let id = ids.pop().ok_or_eyre("no ids")?;
            // remove parent from the parent_to_children map
            parent_to_children.remove(&id).ok_or_eyre("no such id")?;

            // because we are building with nodes that have no children, we can safely remove the current trace from the node_entries_by_id
            let (parent_id, vm_trace_hash, trace) = node_entries_by_id.remove(&id).ok_or_eyre("no such id")?;            
            if let Some(parent_id) = parent_id {
                // remove child from parent
                parent_to_children.get_mut(&parent_id).ok_or_eyre("no parent id")?.remove(&id);
                // if the parent has no children, add it to the ids list
                if parent_to_children.get(&parent_id).ok_or_eyre("no parent id")?.len() == 0 {
                    ids.push(parent_id);
                }
                // becase we start nodes with no children, so we don't expect the parent to be not found.
                node_entries_by_id.get_mut(&parent_id).ok_or_eyre("no parent id")?.1.children.push(vm_trace_hash);
                node_entries_by_id.get_mut(&parent_id).ok_or_eyre("no parent id")?.2.children.push(trace);
            } else {
                // if this is the root trace, set it
                root_trace = Some(trace);
                root_vm_trace_hash = Some(vm_trace_hash);
            }
        }

        info!(
            "[heimdall] build_all_traces: route_len={} simple_cfg={} queue_iterations={} segment_count={} branch_count={} duration_ms={}",
            route_len,
            simple_cfg,
            queue_iterations,
            segment_count,
            branch_count,
            build_all_traces_start.elapsed().as_millis()
        );

        Ok(Some((root_trace.ok_or_eyre("no root trace")?, root_vm_trace_hash.ok_or_eyre("no root vm trace hash")?)))
    }

    pub fn build_all_traces_selector(
        &mut self,
        selector: &str,
        entry_point: u128,
        branch_limit: Option<u32>,
        segment_limit: Option<u32>,
        loop_limit: Option<u32>,
        simple_cfg: bool,
        route: Option<Vec<(usize, usize)>>,
        processed_nodes: &mut HashSet<U256>,
    ) -> Result<Option<(VMTrace, VMTraceExtended)>> {
        self.calldata = decode_hex(selector)?;

        // step through the bytecode until we reach the entry point
        while self.bytecode.len() >= self.instruction as usize && (self.instruction <= entry_point)
        {
            self.step()?;

            // this shouldn't be necessary, but it's safer to have it
            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        self.build_all_traces(branch_limit, segment_limit, loop_limit, simple_cfg, route, processed_nodes)
    }

    /// Build a concrete trace path for the suffix route, after fast-forwarding the VM through a matched prefix route.
    /// `prefix_route` is only used to proceed the VM to the target status,
    /// while `suffix_route` is the part that is actually returned as the trace path.
    /// This avoids the memory pressure and long execution time caused by building a long trace paths.
    pub fn build_trace_from_partial_route(
        &mut self,
        selector: &str,
        entry_point: u128,
        prefix_route: Vec<(usize, usize)>,
        suffix_route: Vec<(usize, usize)>,
    ) -> Result<(VMTrace, VMTraceExtended)> {
        let build_trace_start = Instant::now();
        let prefix_route_len = prefix_route.len();
        let suffix_route_len = suffix_route.len();
        if suffix_route.is_empty() {
            return Err(eyre::eyre!(
                "suffix route is empty for build_trace_from_partial_route"
            ));
        }

        self.calldata = decode_hex(selector)?;

        // step through the bytecode until we reach the entry point
        while self.bytecode.len() >= self.instruction as usize && (self.instruction <= entry_point)
        {
            self.step()?;
            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        let jumpdest_pc = Self::program_counter(self.bytecode.clone())
            .iter()
            .filter(|(_, v)| v.code == 0x5b)
            .map(|(k, _)| k.clone())
            .collect::<HashSet<U256>>();

        // fast-forward the VM by the prefix route, and one of the next traces must match the suffix route start pc
        let mut current_vm = self.clone();
        if !prefix_route.is_empty() {
            let (_, next_traces, _) =
                current_vm.build_trace_start_from_route(prefix_route, &jumpdest_pc)?;
            current_vm = next_traces
                .into_iter()
                .find(|candidate| candidate.instruction as usize - 1 == suffix_route[0].0)
                .ok_or_else(|| {
                    eyre::eyre!(
                        "failed to fast-forward to suffix route start pc {} after prefix",
                        suffix_route[0].0
                    )
                })?;
        }

        let (trace, trace_ext) = current_vm.build_trace_from_route(suffix_route, &jumpdest_pc)?;
        info!(
            "[heimdall] build_trace_from_partial_route: selector=0x{} entry_point={} prefix_route_len={} suffix_route_len={} duration_ms={}",
            selector,
            entry_point,
            prefix_route_len,
            suffix_route_len,
            build_trace_start.elapsed().as_millis(),
        );
        Ok((trace, trace_ext))
    }

    pub fn build_trace_from_route(
        &mut self,
        route: Vec<(usize, usize)>,
        jumpdest_pc: &HashSet<U256>,
    ) -> Result<(VMTrace, VMTraceExtended)> {
        let build_trace_start = Instant::now();
        let route_len = route.len();
        if route.is_empty() {
            return Err(eyre::eyre!("route is empty"));
        }
        if route[0].0 != self.instruction as usize - 1 {
            return Err(eyre::eyre!("route does not start at the current instruction [path]"));
        }

        let next_possible_segment_hashes_fn = |trace: &VMTrace| -> Result<HashSet<U256>> {
            let mut hashes = HashSet::new();
            let opcode = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.opcode as u128;
            let last_instruction = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.instruction;
            match opcode {
                0x57_u128 => {
                    hashes.insert(Self::jump_stack_hash_helper(jumpdest_pc,
                        trace.operations.last().ok_or_eyre("no operations")?.last_instruction.inputs[0].as_u128() + 1, 
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                    hashes.insert(Self::jump_stack_hash_helper(jumpdest_pc,
                        last_instruction + 1,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
                0x56_u128 => {
                    hashes.insert(Self::jump_stack_hash_helper(jumpdest_pc,
                        trace.operations.last().ok_or_eyre("no operations")?.last_instruction.inputs[0].as_u128() + 1, 
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
                0x5f_u128 | 0x60_u128 | 0x61_u128 | 0x62_u128 | 0x63_u128 | 0x64_u128 | 0x65_u128 |
                0x66_u128 | 0x67_u128 | 0x68_u128 | 0x69_u128 | 0x6a_u128 | 0x6b_u128 | 0x6c_u128 |
                0x6d_u128 | 0x6e_u128 | 0x6f_u128 | 0x70_u128 | 0x71_u128 | 0x72_u128 | 0x73_u128 |
                0x74_u128 | 0x75_u128 | 0x76_u128 | 0x77_u128 | 0x78_u128 | 0x79_u128 | 0x7a_u128 |
                0x7b_u128 | 0x7c_u128 | 0x7d_u128 | 0x7e_u128 | 0x7f_u128 => {
                    let next_instruction = last_instruction + (opcode as u128 - 0x5f) + 1;
                    let hash = Self::jump_stack_hash_helper(jumpdest_pc,
                        next_instruction,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack);                    
                    hashes.insert(hash);
                }
                _ => {
                    hashes.insert(Self::jump_stack_hash_helper(jumpdest_pc,
                        last_instruction + 1,
                        &trace.operations.last().ok_or_eyre("no operations")?.stack));
                }
            }
            Ok(hashes)
        };

        let mut path_nodes: Vec<(VMTrace, U256, HashSet<U256>)> = Vec::new();
        let mut path_profile = PathTraceProfile::default();
        let mut trim_profile = TrimTraceProfile::default();
        let root_node_id = Self::generate_safe_node_id_by_route(route.clone());

        let mut current_vm = self.clone();
        let mut idx = 0usize;
        while idx < route.len() {
            // debug log
            let should_log_progress = idx == 0 || idx + 1 == route.len() || idx % 100 == 0;
            let vm_profile_before =
                if should_log_progress { Some(vm_snapshot_profile(&current_vm)) } else { None };
            
            let current_trace_hash = Self::jump_stack_hash_helper(
                jumpdest_pc,
                current_vm.instruction,
                &current_vm.stack,
            );

            // build the trace
            let segment_build_trace_start = Instant::now();
            let (mut trace, next_traces) = current_vm.build_trace()?;
            let segment_build_trace_duration = segment_build_trace_start.elapsed();
            
            // check whether the trace matches the route
            let last_pc = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.instruction as usize - 1;
            if trace.instruction as usize - 1 != route[idx].0 || last_pc != route[idx].1 {
                return Err(eyre::eyre!(
                    "route segment mismatch at index {}: expected {:?}, got ({}, {})",
                    idx,
                    route[idx],
                    trace.instruction.saturating_sub(1),
                    last_pc,
                ));
            }

            // debug log
            let segment_profile = segment_trace_profile(&trace);
            accumulate_path_trace_profile(&mut path_profile, &segment_profile);
            if let Some(vm_profile_before) = vm_profile_before {
                info!(
                    "[heimdall] build_trace_path_from_route progress: idx={} route_len={} current_pc={} seg_build_trace_ms={} seg_ms_per_op={:.3} seg_ops={} seg_state_mem_bytes={} seg_state_stack_frames={} seg_state_storage_slots={} seg_state_transient_slots={} seg_max_state_mem_bytes={} seg_max_state_stack_frames={} seg_max_state_storage_slots={} seg_max_state_transient_slots={} seg_max_input_op_depth={} seg_max_output_op_depth={} seg_max_stack_op_depth={} accum_segments={} accum_ops={} accum_state_mem_bytes={} accum_state_stack_frames={} accum_state_storage_slots={} accum_state_transient_slots={} accum_event_count={} accum_max_seg_ops={} accum_max_seg_mem_bytes={} accum_max_state_mem_bytes={} accum_max_state_stack_frames={} accum_max_state_storage_slots={} accum_max_state_transient_slots={} accum_max_input_op_depth={} accum_max_output_op_depth={} accum_max_stack_op_depth={} vm_stack_frames_before={} vm_stack_max_op_depth_before={} vm_memory_bytes_before={} vm_storage_slots_before={} vm_transient_slots_before={} vm_event_count_before={} next_candidates={} peak_rss_kb={:?} elapsed_ms={}",
                    idx,
                    route_len,
                    current_vm.instruction.saturating_sub(1),
                    segment_build_trace_duration.as_millis(),
                    segment_build_trace_duration.as_secs_f64() * 1000.0
                        / segment_profile.op_count.max(1) as f64,
                    segment_profile.op_count,
                    segment_profile.total_state_memory_bytes,
                    segment_profile.total_state_stack_frames,
                    segment_profile.total_state_storage_slots,
                    segment_profile.total_state_transient_slots,
                    segment_profile.max_state_memory_bytes,
                    segment_profile.max_state_stack_frames,
                    segment_profile.max_state_storage_slots,
                    segment_profile.max_state_transient_slots,
                    segment_profile.max_state_input_op_depth,
                    segment_profile.max_state_output_op_depth,
                    segment_profile.max_state_stack_op_depth,
                    path_profile.segment_count,
                    path_profile.total_ops,
                    path_profile.total_state_memory_bytes,
                    path_profile.total_state_stack_frames,
                    path_profile.total_state_storage_slots,
                    path_profile.total_state_transient_slots,
                    path_profile.total_state_event_count,
                    path_profile.max_segment_ops,
                    path_profile.max_segment_total_state_memory_bytes,
                    path_profile.max_segment_max_state_memory_bytes,
                    path_profile.max_segment_max_state_stack_frames,
                    path_profile.max_segment_max_state_storage_slots,
                    path_profile.max_segment_max_state_transient_slots,
                    path_profile.max_input_op_depth,
                    path_profile.max_output_op_depth,
                    path_profile.max_stack_op_depth,
                    vm_profile_before.stack_frames,
                    vm_profile_before.stack_max_op_depth,
                    vm_profile_before.memory_bytes,
                    vm_profile_before.storage_slots,
                    vm_profile_before.transient_slots,
                    vm_profile_before.event_count,
                    next_traces.len(),
                    process_peak_rss_kb(),
                    build_trace_start.elapsed().as_millis(),
                );
            }

            let next_possible_segment_hashes = next_possible_segment_hashes_fn(&trace)
                .map_err(|e| eyre::eyre!("failed to get next possible segment hashes: {}", e))?;
            
            // test
            // let segment_trim_profile = trim_trace_for_storage(&mut trace);
            // trim_profile.original_op_count += segment_trim_profile.original_op_count;
            // trim_profile.retained_op_count += segment_trim_profile.retained_op_count;
            // trim_profile.removed_op_count += segment_trim_profile.removed_op_count;
            // trim_profile.cleared_stack_frames += segment_trim_profile.cleared_stack_frames;

            // find the next trace that matches the next route start pc
            if let Some(next_segment) = route.get(idx + 1) {
                let next_vm = next_traces
                    .into_iter()
                    .find(|candidate| candidate.instruction as usize - 1 == next_segment.0)
                    .ok_or_else(|| {
                        eyre::eyre!(
                            "failed to follow execution route at index {}: next segment start pc {} not found",
                            idx,
                            next_segment.0
                        )
                    })?;
                path_nodes.push((trace, current_trace_hash, next_possible_segment_hashes));
                current_vm = next_vm;
            } else {
                path_nodes.push((trace, current_trace_hash, next_possible_segment_hashes));
            }

            idx += 1;
        }

        let assembly_path_nodes_len = path_nodes.len();
        let assembly_start = Instant::now();
        let mut current_trace: Option<VMTrace> = None;
        let mut current_trace_extended: Option<VMTraceExtended> = None;
        for (offset, (mut trace, hash, next_possible_segment_hashes)) in
            path_nodes.into_iter().enumerate().rev()
        {
            if let Some(child_trace) = current_trace.take() {
                trace.children.push(child_trace);
            }

            let mut trace_extended = VMTraceExtended {
                id: root_node_id + offset as u32,
                hash,
                children: Vec::new(),
                next_possible_segment_hashes,
            };
            if let Some(child_trace_extended) = current_trace_extended.take() {
                trace_extended.children.push(child_trace_extended);
            }

            current_trace = Some(trace);
            current_trace_extended = Some(trace_extended);
        }
        let assembly_duration = assembly_start.elapsed();
        debug!(
            "[heimdall] build_trace_from_route assembly: route_len={} path_nodes_len={} assembly_duration_ms={} assembly_ms_per_node={:.3} peak_rss_kb={:?}",
            route_len,
            assembly_path_nodes_len,
            assembly_duration.as_millis(),
            assembly_duration.as_secs_f64() * 1000.0 / assembly_path_nodes_len.max(1) as f64,
            process_peak_rss_kb(),
        );

        debug!(
            "[heimdall] build_trace_from_route: route_len={} duration_ms={}",
            route_len,
            build_trace_start.elapsed().as_millis()
        );

        info!(
            "[heimdall] build_trace_path_from_route summary: route_len={} built_nodes={} total_ops={} total_state_mem_bytes={} total_state_stack_frames={} total_state_storage_slots={} total_state_transient_slots={} total_event_count={} max_seg_ops={} max_seg_total_state_mem_bytes={} max_state_mem_bytes={} max_state_stack_frames={} max_state_storage_slots={} max_state_transient_slots={} max_input_op_depth={} max_output_op_depth={} max_stack_op_depth={} trimmed_original_ops={} trimmed_retained_ops={} trimmed_removed_ops={} trimmed_cleared_stack_frames={} peak_rss_kb={:?} duration_ms={}",
            route_len,
            route_len,
            path_profile.total_ops,
            path_profile.total_state_memory_bytes,
            path_profile.total_state_stack_frames,
            path_profile.total_state_storage_slots,
            path_profile.total_state_transient_slots,
            path_profile.total_state_event_count,
            path_profile.max_segment_ops,
            path_profile.max_segment_total_state_memory_bytes,
            path_profile.max_segment_max_state_memory_bytes,
            path_profile.max_segment_max_state_stack_frames,
            path_profile.max_segment_max_state_storage_slots,
            path_profile.max_segment_max_state_transient_slots,
            path_profile.max_input_op_depth,
            path_profile.max_output_op_depth,
            path_profile.max_stack_op_depth,
            trim_profile.original_op_count,
            trim_profile.retained_op_count,
            trim_profile.removed_op_count,
            trim_profile.cleared_stack_frames,
            process_peak_rss_kb(),
            build_trace_start.elapsed().as_millis(),
        );

        Ok((
            current_trace.ok_or_eyre("no root trace")?,
            current_trace_extended.ok_or_eyre("no root vm trace extended")?,
        ))
    }

    // build a trace starting from a given route
    // each element in the route is a tuple of (start pc in segment, end pc in segment)
    // if the segment only has one instruction, the end pc should be the same as the start pc.
    pub fn build_trace_start_from_route(&mut self, mut route: Vec<(usize, usize)>, jumpdest_pc: &HashSet<U256>) -> Result<(VMTrace, Vec<VM>, HashMap<U256, u32>)> {
        let build_trace_start_from_route_start = Instant::now();
        let route_len = route.len();
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

        let mut next_traces = Vec::new();
        let mut route_segment_hashes: HashMap<U256, u32> = HashMap::new();
        let mut step_count = 0usize;
        let mut last_sample_ms = 0f64;
        let mut total_jump_hash_ms = 0f64;
        let mut jump_hash_call_count = 0usize;
        {
            let snap = vm_snapshot_profile(self);
            let rss_kb = process_peak_rss_kb().unwrap_or(0);
            debug!(
                "[heimdall] build_trace_start_from_route init: route_len={} rss_kb={} \
                 stack_frames={} stack_max_op_depth={} memory_bytes={} storage_slots={}",
                route_len, rss_kb,
                snap.stack_frames, snap.stack_max_op_depth,
                snap.memory_bytes, snap.storage_slots,
            );
        }
        while self.bytecode.len() >= self.instruction as usize {
            step_count += 1;
            let state = self.step()?;
            let last_instruction = state.last_instruction.clone();

            if step_count % 1000 == 0 {
                let elapsed_ms = build_trace_start_from_route_start.elapsed().as_secs_f64() * 1000.0;
                let snap = vm_snapshot_profile(self);
                let rss_kb = process_peak_rss_kb().unwrap_or(0);
                let interval_ms = elapsed_ms - last_sample_ms;
                let jump_hash_pct = if elapsed_ms > 0.0 { total_jump_hash_ms / elapsed_ms * 100.0 } else { 0.0 };
                last_sample_ms = elapsed_ms;
                debug!(
                    "[heimdall] build_trace_start_from_route sample: step={} elapsed_ms={:.0} \
                     interval_ms={:.0} rss_kb={} \
                     stack_frames={} stack_max_op_depth={} stack_total_op_nodes={} \
                     memory_bytes={} memory_op_entries={} memory_max_op_depth={} memory_total_op_nodes={} \
                     storage_slots={} jump_hash_calls={} total_jump_hash_ms={:.0} jump_hash_pct={:.1}%",
                    step_count, elapsed_ms, interval_ms, rss_kb,
                    snap.stack_frames, snap.stack_max_op_depth, snap.stack_total_op_nodes,
                    snap.memory_bytes, snap.memory_op_entries, snap.memory_max_op_depth, snap.memory_total_op_nodes,
                    snap.storage_slots,
                    jump_hash_call_count, total_jump_hash_ms, jump_hash_pct,
                );
            }

            if last_instruction.opcode == 0x57 { // jumpi
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        self.instruction = last_instruction.instruction + 1;
                        next_traces.push(self.clone());
                        self.instruction = last_instruction.inputs[0].as_u128() + 1;
                        next_traces.push(self.clone());
                        break;
                    } else {
                        current_node = route.pop().ok_or_eyre("no route")?;
                        if current_node.0 == last_instruction.instruction as usize {
                            self.instruction = last_instruction.instruction + 1;
                        } else if current_node.0 == last_instruction.inputs[0].as_u128() as usize {
                            self.instruction = last_instruction.inputs[0].as_u128() + 1;
                        } else {
                            return Err(eyre::eyre!("route does not match the last instruction [2]"));
                        }
                        let _t = Instant::now();
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        total_jump_hash_ms += _t.elapsed().as_secs_f64() * 1000.0;
                        jump_hash_call_count += 1;
                        *route_segment_hashes.entry(seg_hash).or_insert(0) += 1;
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [3]"));
                }
            } else if last_instruction.opcode == 0x56 { // jump
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
                        } else {
                            return Err(eyre::eyre!("route does not match the last instruction [4]"));
                        }
                        let _t = Instant::now();
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        total_jump_hash_ms += _t.elapsed().as_secs_f64() * 1000.0;
                        jump_hash_call_count += 1;
                        *route_segment_hashes.entry(seg_hash).or_insert(0) += 1;
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [5]"));
                }
            } else if last_instruction.opcode == 0x00 ||
                    last_instruction.opcode == 0xfd ||
                    last_instruction.opcode == 0xfe ||
                    last_instruction.opcode == 0xff ||
                    last_instruction.opcode == 0xf3 { // if meet the stop, revert, invalid, selfdestruct, or return instruction, we should end the trace
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
                        return Err(eyre::eyre!("route should end here, but got more instructions in the route"));
                    }
                } else {
                    return Err(eyre::eyre!("route does not match the last instruction [8]"));
                }
            } else if self
                .bytecode
                .get((self.instruction - 1) as usize)
                .ok_or_eyre(format!("invalid jumpdest: {}", self.instruction - 1))?
            .to_owned() == 0x5b {                    
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
                        let _t = Instant::now();
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
                        total_jump_hash_ms += _t.elapsed().as_secs_f64() * 1000.0;
                        jump_hash_call_count += 1;
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

        let duration_ms = build_trace_start_from_route_start.elapsed().as_secs_f64() * 1000.0;
        let ms_per_step = if step_count == 0 {
            0.0
        } else {
            duration_ms / step_count as f64
        };
        debug!(
            "[heimdall] build_trace_start_from_route: route_len={} step_count={} next_traces_len={} duration_ms={:.3} ms_per_step={:.6}",
            route_len,
            step_count,
            next_traces.len(),
            duration_ms,
            ms_per_step
        );

        Ok((root_trace, next_traces, route_segment_hashes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_vm(bytecode: &[u8]) -> VM {
        let mut vm = VM::new(
            bytecode,
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
            0,
            u128::MAX,
        );
        vm.storage = Default::default();
        vm.stack = Default::default();
        vm.memory = Default::default();
        vm.instruction = 1;
        vm
    }

    fn collect_route(trace: &VMTrace) -> Vec<(usize, usize)> {
        let mut route = Vec::new();
        let mut current = trace;
        loop {
            let last_pc = current
                .operations
                .last()
                .expect("trace has operations")
                .last_instruction
                .instruction as usize
                - 1;
            route.push((current.instruction as usize - 1, last_pc));

            if current.children.is_empty() {
                break;
            }
            assert_eq!(
                current.children.len(),
                1,
                "path trace should stay linear while collecting route"
            );
            current = &current.children[0];
        }
        route
    }

    fn collect_extended_ids(trace: &VMTraceExtended) -> Vec<u32> {
        let mut ids = vec![trace.id];
        let mut current = trace;
        while !current.children.is_empty() {
            assert_eq!(
                current.children.len(),
                1,
                "extended path trace should stay linear while collecting ids"
            );
            current = &current.children[0];
            ids.push(current.id);
        }
        ids
    }

    #[test]
    fn build_trace_path_from_route_follows_the_concrete_branch() {
        let bytecode = heimdall_common::utils::strings::decode_hex("60016008576002005b600300")
            .expect("valid bytecode");
        let route = vec![(0usize, 4usize), (8usize, 11usize)];
        let jumpdest_pc = VM::program_counter(bytecode.clone())
            .iter()
            .filter(|(_, opcode)| opcode.code == 0x5b)
            .map(|(pc, _)| *pc)
            .collect::<HashSet<U256>>();

        let mut vm = build_test_vm(&bytecode);
        let (trace, trace_ext) = vm
            .build_trace_from_route(route.clone(), &jumpdest_pc)
            .expect("route should materialize into a linear trace path");

        assert_eq!(collect_route(&trace), route);

        let extended_ids = collect_extended_ids(&trace_ext);
        assert_eq!(extended_ids.len(), route.len());
        assert_eq!(
            extended_ids,
            (extended_ids[0]..extended_ids[0] + route.len() as u32).collect::<Vec<_>>()
        );
        assert!(trace_ext.next_possible_segment_hashes.len() >= 2);
        assert_eq!(
            trace_ext.children.first().map(|child| child.next_possible_segment_hashes.len()),
            Some(1)
        );
    }
}
