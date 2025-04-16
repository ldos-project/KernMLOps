
# void run_on_stack(void* new_stack, void (*work)(void))
    .globl run_on_stack
    .p2align	4
    .type	run_on_stack,@function
run_on_stack:
    push %rbp
    mov %rsp, %rdx # rdx = kernel_sp
    mov %rdi, %rsp # sp = new_stack
    push %rdx
    mov %rsp, %rbp
    mov %rsi, %rax # call work
    call *%rax #call __x86_indirect_thunk_rax
    pop %rdx
    mov %rdx, %rsp # restore kernel_sp
    pop %rbp
    ret 

.section .note.GNU-stack,"",@progbits
    
