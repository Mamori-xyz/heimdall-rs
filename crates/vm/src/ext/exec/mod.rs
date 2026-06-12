mod jump_frame;
mod util;

use std::{cell::RefCell, sync::Arc};
use ethers::abi::AbiEncode;
use ethers::prelude::U256;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::{
    core::{
        stack::Stack,
        vm::{Instruction, State, VM},
        opcodes::Opcode
    },
    ext::exec::{
        jump_frame::JumpFrame,
        util::{
            historical_diffs_approximately_equal, jump_condition_appears_recursive,
            jump_condition_contains_mutated_memory_access,
            jump_condition_contains_mutated_storage_access,
            jump_stack_depth_less_than_max_stack_depth, stack_contains_too_many_items,
            stack_contains_too_many_of_the_same_item, stack_diff, stack_item_source_depth_too_deep,
        },
    },
};
use eyre::{OptionExt, Result};
use heimdall_common::utils::strings::decode_hex;
use std::{collections::HashMap, time::Instant};
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
    /// Set when this segment ends in a JUMPI whose condition folds to an input-independent
    /// constant. The CFG builder explores only the feasible successor; this records the decision
    /// so downstream consumers (e.g. the CFG viewer) can flag that the other direction is a
    /// statically dead end without re-deriving it. `None` for any non-static or non-JUMPI segment.
    pub static_jumpi: Option<StaticJumpi>,
}

/// Records a statically-resolved JUMPI: the branch the CFG builder pruned and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticJumpi {
    /// The proven-constant condition value the JUMPI folded to.
    pub condition_value: U256,
    /// `true`  => the jump target is the live branch (fall-through was pruned).
    /// `false` => fall-through is the live branch (the jump target was pruned).
    pub jump_taken: bool,
    /// EVM PC of the JUMPI's jump target (the JUMPDEST it would land on if taken). Always known
    /// because the target is a concrete value on the stack.
    pub jump_target_pc: u128,
    /// EVM PC of the pruned, statically-unreachable successor's first instruction. For a viewer
    /// this is the node that was *not* generated, i.e. where the dead-end marker belongs.
    pub dead_successor_pc: u128,
}

/// Determine the statically-known outcome of a JUMPI from its last instruction.
///
/// At CFG-build time the VM already holds both the concrete condition value (`inputs[1]`) and the
/// symbolic expression that produced it (`input_operations[1]`). The concrete value only reflects
/// whatever placeholder calldata the build started with, so it is a *true* static constant only when
/// the symbolic expression depends on no calldata/storage/memory/environment opcode. When that holds,
/// the concrete value is the static condition and we can drop the infeasible successor entirely.
///
/// - `Some(true)`  => condition is a non-zero constant, JUMPI is always taken (only the jump target
///   is feasible).
/// - `Some(false)` => condition is a zero constant, JUMPI is never taken (only fall-through is
///   feasible).
/// - `None`        => condition is not a proven constant; both successors must be kept.
fn jumpi_static_outcome(last_instruction: &Instruction) -> Option<bool> {
    let condition = last_instruction.input_operations.get(1)?;
    if !condition.is_constant() {
        return None;
    }
    Some(!last_instruction.inputs.get(1)?.is_zero())
}

/// Build the [`StaticJumpi`] flag for a finished segment, if it ends in a statically-resolved JUMPI.
/// Returns `None` for non-JUMPI segments and for JUMPIs whose condition is not a proven constant.
fn static_jumpi_of_trace(trace: &VMTrace) -> Option<StaticJumpi> {
    let last_instruction = &trace.operations.last()?.last_instruction;
    if last_instruction.opcode != 0x57 {
        return None;
    }
    let jump_taken = jumpi_static_outcome(last_instruction)?;
    let condition_value = *last_instruction.inputs.get(1)?;
    // EVM PC of each successor's first instruction (heimdall pc == 1-indexed instruction pointer):
    //   jump target  -> the JUMPDEST value pushed for the jump (inputs[0])
    //   fall-through -> the instruction immediately after the JUMPI (its 1-indexed pointer)
    let jump_target_pc = last_instruction.inputs.get(0)?.as_u128();
    let fall_through_pc = last_instruction.instruction;
    let dead_successor_pc = if jump_taken { fall_through_pc } else { jump_target_pc };
    Some(StaticJumpi { condition_value, jump_taken, jump_target_pc, dead_successor_pc })
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
        let mut hash_data: Vec<U256> = Vec::new();
        hash_data.push(U256::from(pc));        
        for (i, s) in stack.stack.iter().enumerate() {        
            let solidified_operation = s.operation.solidify();
            if jump_dest_pc.contains(&s.value) && solidified_operation.starts_with("0x") && !solidified_operation.contains(" ") {
                hash_data.push(U256::from(i));
                hash_data.push(s.value);                            
            }            
        }
    
        let mut data: Vec<u8> = Vec::new();
        for v in &hash_data {
            data.append(&mut v.to_string().into_bytes());
        }
        let jump_and_jumpi_hash = U256::from(ethers::core::utils::keccak256(&data));
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
            let hash_within_u32= hash % U256::from(u32::MAX);
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
                    // JUMPI: if the condition folds to an input-independent constant, only the
                    // feasible successor is generated and the unreachable branch is omitted
                    // entirely (no successor VM => no node/edge/subtree). Otherwise both branches
                    // are explored, preserving the original (continue-first, jump-second) order.
                    match jumpi_static_outcome(&last_instruction) {
                        Some(true) => {
                            // always taken: only the jump target is reachable
                            let mut new_trace = self.clone();
                            new_trace.instruction = last_instruction.inputs[0].as_u128() + 1;
                            next_traces.push(new_trace);
                        }
                        Some(false) => {
                            // never taken: only the fall-through is reachable
                            let mut new_trace = self.clone();
                            new_trace.instruction = last_instruction.instruction + 1;
                            next_traces.push(new_trace);
                        }
                        None => {
                            // continue branch
                            let mut continue_trace = self.clone();
                            continue_trace.instruction = last_instruction.instruction + 1;
                            next_traces.push(continue_trace);

                            // jump branch
                            let mut jump_trace = self.clone();
                            jump_trace.instruction = last_instruction.inputs[0].as_u128() + 1;
                            next_traces.push(jump_trace);
                        }
                    }
                    break;
                }

                // unconditional JUMP (0x56): single successor at the jump target
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
        let root_node_id = node_counter;
        
        let next_possible_segment_hashes_fn = |trace: &VMTrace| -> Result<HashSet<U256>> {
            let mut hashes = HashSet::new();
            let opcode = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.opcode as u128;
            let last_instruction = trace.operations.last().ok_or_eyre("no operations")?.last_instruction.instruction;
            match opcode {
                0x57_u128 => {
                    // Mirror the static-pruning gate applied in `build_trace`: when the JUMPI
                    // condition is a proven constant, only the feasible successor's hash is listed,
                    // so `next_possible_segment_hashes` stays consistent with the children actually
                    // generated.
                    let op = trace.operations.last().ok_or_eyre("no operations")?;
                    let jump_target = op.last_instruction.inputs[0].as_u128() + 1;
                    let fall_through = last_instruction + 1;
                    match jumpi_static_outcome(&op.last_instruction) {
                        Some(true) => {
                            hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc, jump_target, &op.stack));
                        }
                        Some(false) => {
                            hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc, fall_through, &op.stack));
                        }
                        None => {
                            hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc, jump_target, &op.stack));
                            hashes.insert(Self::jump_stack_hash_helper(&jumpdest_pc, fall_through, &op.stack));
                        }
                    }
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
                static_jumpi: static_jumpi_of_trace(&root_trace),
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
            if queue_iterations % 1000 == 0 {
                debug!(
                    "[heimdall] build_all_traces debug: route_len={} simple_cfg={} queue_iterations={} queue_size={} segment_count={} branch_count={} processed_nodes={} previous_trace_hash={} elapsed_ms={}",
                    route_len, simple_cfg, queue_iterations, queue.len(),
                    segment_count, branch_count, processed_nodes.len(),
                    previous_trace_hash.len(),
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
            let (trace, mut next_traces) = vm.build_trace()?;

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
            node_entries_by_id.insert(node_counter,
                (Some(parent_id),
                    VMTraceExtended {
                        id: node_counter,
                        hash: current_trace_hash,
                        children: Vec::new(),
                        next_possible_segment_hashes: next_possible_segment_hashes_fn(&trace).map_err(|e| eyre::eyre!("failed to get next possible segment hashes: {}", e))?,
                        static_jumpi: static_jumpi_of_trace(&trace),
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
        debug!(
            "[heimdall] build_all_traces: route_len={} simple_cfg={} queue_iterations={} segment_count={} branch_count={} duration_ms={}",
            route_len,
            simple_cfg,
            queue_iterations,
            segment_count,
            branch_count,
            queue_expand_start.elapsed().as_millis()
        );

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
        while self.bytecode.len() >= self.instruction as usize {    
            step_count += 1;
            let state = self.step()?;
            let last_instruction = state.last_instruction.clone();
            
            if last_instruction.opcode == 0x57 { // jumpi
                if last_instruction.instruction as usize - 1 == current_node.1 {
                    if route.is_empty() {
                        root_trace = VMTrace {
                            instruction: last_instruction.instruction,
                            gas_used: self.gas_used,
                            operations: Vec::from([state]),
                            children: Vec::new(),
                        };
                        // The route ends exactly on this JUMPI, so both directions are unexplored
                        // frontier. Apply the same static-pruning gate as the off-route expansion:
                        // a constant condition means only one side is feasible, so we extend toward
                        // it alone and never expand the unreachable branch.
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
                            self.instruction = last_instruction.instruction + 1;
                        } else if current_node.0 == last_instruction.inputs[0].as_u128() as usize {
                            self.instruction = last_instruction.inputs[0].as_u128() + 1;
                        } else {
                            return Err(eyre::eyre!("route does not match the last instruction [2]"));
                        }
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
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
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
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
                        let seg_hash = Self::jump_stack_hash_helper(jumpdest_pc, self.instruction, &self.stack);
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
    use crate::core::vm::VM;
    use ethers::types::H160;

    // Build a VM over `bytecode` (empty calldata) and run `build_trace`, returning the number of
    // successor VMs the fork produced.
    fn successor_count(bytecode: &[u8]) -> usize {
        let mut vm = VM::new(
            bytecode,
            &[],
            H160::zero(),
            H160::zero(),
            H160::zero(),
            0,
            1_000_000,
        );
        let (_trace, next_traces) = vm.build_trace().expect("build_trace failed");
        next_traces.len()
    }

    #[test]
    fn test_build_trace_prunes_constant_false_jumpi() {
        // PUSH1 0x00 (cond=0); PUSH1 0x07 (dest); JUMPI; STOP; pad; JUMPDEST; STOP
        // condition is a zero constant => never taken => only the fall-through successor.
        let bytecode = [0x60, 0x00, 0x60, 0x07, 0x57, 0x00, 0x00, 0x5b, 0x00];
        assert_eq!(successor_count(&bytecode), 1);
    }

    #[test]
    fn test_build_trace_prunes_constant_true_jumpi() {
        // PUSH1 0x01 (cond=1); PUSH1 0x07 (dest); JUMPI; STOP; pad; JUMPDEST; STOP
        // condition is a non-zero constant => always taken => only the jump-target successor.
        let bytecode = [0x60, 0x01, 0x60, 0x07, 0x57, 0x00, 0x00, 0x5b, 0x00];
        assert_eq!(successor_count(&bytecode), 1);
    }

    #[test]
    fn test_build_trace_keeps_both_for_dynamic_jumpi() {
        // PUSH1 0x00; CALLDATALOAD (cond depends on input); PUSH1 0x08 (dest); JUMPI; ...
        // condition is not a proven constant => both successors are explored.
        let bytecode = [0x60, 0x00, 0x35, 0x60, 0x08, 0x57, 0x00, 0x00, 0x5b, 0x00];
        assert_eq!(successor_count(&bytecode), 2);
    }

    // Collect every StaticJumpi flag in a VMTraceExtended tree.
    fn collect_static_jumpis(ext: &super::VMTraceExtended) -> Vec<super::StaticJumpi> {
        let mut out = Vec::new();
        let mut stack = vec![ext];
        while let Some(node) = stack.pop() {
            if let Some(sj) = &node.static_jumpi {
                out.push(sj.clone());
            }
            for child in &node.children {
                stack.push(child);
            }
        }
        out
    }

    fn build_all(bytecode: &[u8]) -> super::VMTraceExtended {
        let mut vm = VM::new(
            bytecode,
            &[],
            H160::zero(),
            H160::zero(),
            H160::zero(),
            0,
            1_000_000,
        );
        let (_trace, ext) = vm
            .build_all_traces(None, None, None, false, None, &mut std::collections::HashSet::new())
            .expect("build_all_traces failed")
            .expect("expected a trace");
        ext
    }

    #[test]
    fn test_static_jumpi_flag_set_for_constant_false() {
        // condition = 0, dest = 7 => never taken, jump target (pc 7) is the dead successor.
        let bytecode = [0x60, 0x00, 0x60, 0x07, 0x57, 0x00, 0x00, 0x5b, 0x00];
        let flags = collect_static_jumpis(&build_all(&bytecode));
        assert_eq!(flags.len(), 1);
        assert!(!flags[0].jump_taken);
        assert_eq!(flags[0].jump_target_pc, 7);
        // never taken => the dead branch is the jump target (pc 7)
        assert_eq!(flags[0].dead_successor_pc, 7);
        assert!(flags[0].condition_value.is_zero());
    }

    #[test]
    fn test_static_jumpi_flag_absent_for_dynamic() {
        // condition from CALLDATALOAD => no static flag anywhere in the tree.
        let bytecode = [0x60, 0x00, 0x35, 0x60, 0x08, 0x57, 0x00, 0x00, 0x5b, 0x00];
        let flags = collect_static_jumpis(&build_all(&bytecode));
        assert!(flags.is_empty());
    }
}
