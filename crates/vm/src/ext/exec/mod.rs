mod jump_frame;
mod util;

use std::{cell::RefCell, sync::Arc};
use ethers::prelude::U256;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::{
    core::{
        stack::Stack,
        vm::{State, VM},
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
use tracing::{trace, warn};

#[derive(Clone, Debug, Default)]
pub struct VMTrace {
    pub instruction: u128,
    pub gas_used: u128,
    pub operations: Vec<State>,
    pub children: Vec<VMTrace>,
}

#[derive(Clone, Debug, Default)]
pub struct VMTraceHash {
    pub hash: U256,
    pub children: Vec<VMTraceHash>
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
        simple_cfg: bool) -> Result<Option<(VMTrace, VMTraceHash)>> {         
        let mut branch_count: u32 = 0;
        let mut segment_count: u32 = 0;

        let jumpdest_pc = Self::program_counter(self.bytecode.clone())
            .iter()
            .filter(|(k, v)| v.code == 0x5b)
            .map(|(k, _)| k.clone())
            .collect::<HashSet<U256>>();

        let mut node_counter: u32 = 0;
        let (root_trace, mut next_traces) = self.build_trace()?;
        let root_trace_hash = Self::jump_stack_hash_helper(&jumpdest_pc,
            root_trace.operations.first().ok_or_eyre("no operations")?.last_instruction.instruction, 
            &root_trace.operations.first().ok_or_eyre("no operations")?.stack);        

        // init the root trace    
        let mut previous_trace_hash = HashMap::new();
        previous_trace_hash.insert(root_trace_hash.clone(), 1);  

        // update the branch and segment counts for the root trace
        branch_count += 1;
        segment_count += 1;

        let mut parent_to_children: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut node_entries_by_id: HashMap<u32, (Option<u32>, VMTraceHash, VMTrace)> = HashMap::new(); // (parent_id, trace_hash, trace)
        node_entries_by_id.insert(node_counter, (None,
            VMTraceHash {
                hash: root_trace_hash,
                children: Vec::new(),
            },
            root_trace,
        ));
        parent_to_children.entry(node_counter).or_insert(HashSet::new());

        // initialize the queue with the first set of traces
        let mut queue: VecDeque<(u32, HashMap<U256, u32>, VM)> = VecDeque::new();
        while !next_traces.is_empty() {        
            queue.push_front((node_counter, previous_trace_hash.clone(), next_traces.pop().ok_or_eyre("no next traces")?));   
        }

        // only used for simple cfg
        let mut processed_nodes = HashSet::new();  
        // process the queue until it is empty
        while !queue.is_empty() {   
            // only check branch and segment limits if we are not building a simple cfg
            if !simple_cfg {
                if branch_limit.is_some() && branch_count >= branch_limit.unwrap() {
                    return Ok(None);
                }
                if segment_limit.is_some() && segment_count >= segment_limit.unwrap() {
                    return Ok(None);
                }
            }            
            
            let (parent_id, mut previous_trace_hash, mut vm) = queue.pop_front().ok_or_eyre("no next traces")?;
            let (trace, mut next_traces) = vm.build_trace()?;
            // validate with loop detection heuristics. if the trace is a loop, skip it
            let current_first_pc = trace.operations.first().ok_or_eyre("no operations")?.last_instruction.instruction;
            let current_trace_hash = Self::jump_stack_hash_helper(&jumpdest_pc,
                current_first_pc, 
                &trace.operations.first().ok_or_eyre("no operations")?.stack);

            // loop detection
            *previous_trace_hash.entry(current_trace_hash).or_insert(0) += 1;
            let loop_limit = loop_limit.unwrap_or(1);
            if *previous_trace_hash.get(&current_trace_hash).ok_or_eyre("no current trace hash")? > loop_limit {
                continue;
            }

            // if we are building a simple cfg, we only want to process each similar trace once by checking globally
            if simple_cfg {     
                if !processed_nodes.contains(&current_trace_hash) {
                    processed_nodes.insert(current_trace_hash);
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
                    VMTraceHash {
                        hash: current_trace_hash,
                        children: Vec::new(),
                    },
                    trace,
                ),
            );
            parent_to_children.entry(parent_id).or_insert(HashSet::new()).insert(node_counter);
            parent_to_children.entry(node_counter).or_insert(HashSet::new());
            while !next_traces.is_empty() {                
                queue.push_front((node_counter, previous_trace_hash.clone(), next_traces.pop().ok_or_eyre("no next traces")?));   
            }
        }        

        // always start with the nodes that have no children
        let mut ids = parent_to_children.iter().filter_map(|(parent_id, children)| {
            if children.is_empty() {
                Some(*parent_id)
            } else {
                None
            }
        }).collect::<Vec<u32>>();

        let mut root_trace: Option<VMTrace> = None;
        let mut root_vm_trace_hash: Option<VMTraceHash> = None;
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
    ) -> Result<Option<(VMTrace, VMTraceHash)>> {
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
        
        self.build_all_traces(branch_limit, segment_limit, loop_limit, simple_cfg)
    }
}

#[cfg(test)]
mod tests {
    // TODO: add tests for symbolic execution & recursive_map
}
