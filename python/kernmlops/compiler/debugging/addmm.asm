    .file   "LLVMDialectModule"
    .text
    .globl  addmm                           # -- Begin function addmm
    .p2align    4
    .type   addmm,@function
addmm:                                  # @addmm
.Lfunc_begin0:
    .file   1 "/home/uzairn/github/KernMLOps/python/kernmlops/compiler/extern_kernels" "addmm.py"
    .loc    1 6 0                           # addmm.py:6:0
    .cfi_sections .debug_frame
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    pushq   %r15
    .cfi_def_cfa_offset 24
    pushq   %r14
    .cfi_def_cfa_offset 32
    pushq   %r13
    .cfi_def_cfa_offset 40
    pushq   %r12
    .cfi_def_cfa_offset 48
    pushq   %rbx
    .cfi_def_cfa_offset 56
    .cfi_offset %rbx, -56
    .cfi_offset %r12, -48
    .cfi_offset %r13, -40
    .cfi_offset %r14, -32
    .cfi_offset %r15, -24
    .cfi_offset %rbp, -16
    movq    %r9, -24(%rsp)                  # 8-byte Spill
    movq    %rcx, -16(%rsp)                 # 8-byte Spill
.Ltmp0:
    .loc    1 18 21 prologue_end            # addmm.py:18:21
    testl   %edi, %edi
    jle .LBB0_13
# %bb.1:                                # %.lr.ph
    .loc    1 0 21 is_stmt 0                # addmm.py:0:21
    testl   %esi, %esi
    jle .LBB0_13
# %bb.2:                                # %.lr.ph.split.us
    movl    120(%rsp), %eax
    movl    104(%rsp), %r11d
    .loc    1 18 21 is_stmt 1               # addmm.py:18:21
    movl    %edi, %ecx
    movq    %rcx, -32(%rsp)                 # 8-byte Spill
    movl    %esi, %esi
    testl   %edx, %edx
    jle .LBB0_3
# %bb.7:                                # %.lr.ph4.us.us.preheader
    .loc    1 0 21 is_stmt 0                # addmm.py:0:21
    movl    80(%rsp), %r15d
    movl    72(%rsp), %r12d
    movl    %edx, %ebp
    xorl    %edi, %edi
    .p2align    4
.LBB0_8:                                # %.lr.ph4.us.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_9 Depth 2
                                        #       Child Loop BB0_10 Depth 3
    .loc    1 22 44 is_stmt 1               # addmm.py:22:44
    movl    64(%rsp), %ecx
    imull   %edi, %ecx
    .loc    1 22 40 is_stmt 0               # addmm.py:22:40
    movslq  %ecx, %rcx
    movq    -16(%rsp), %rdx                 # 8-byte Reload
    leaq    (%rdx,%rcx,4), %r14
    .loc    1 25 40 is_stmt 1               # addmm.py:25:40
    movl    96(%rsp), %ecx
    imull   %edi, %ecx
    .loc    1 25 36 is_stmt 0               # addmm.py:25:36
    movslq  %ecx, %rcx
    movq    -24(%rsp), %rdx                 # 8-byte Reload
    leaq    (%rdx,%rcx,4), %r13
    .loc    1 26 35 is_stmt 1               # addmm.py:26:35
    movl    112(%rsp), %ecx
    movq    %rdi, -8(%rsp)                  # 8-byte Spill
    imull   %edi, %ecx
    .loc    1 26 31 is_stmt 0               # addmm.py:26:31
    movslq  %ecx, %rcx
    movq    56(%rsp), %rdx
    leaq    (%rdx,%rcx,4), %rbx
    xorl    %edi, %edi
    .loc    1 0 31                          # :0:31
.Ltmp1:
    .p2align    4
.LBB0_9:                                # %.lr.ph.us.us.us
                                        #   Parent Loop BB0_8 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_10 Depth 3
    .loc    1 23 58 is_stmt 1               # addmm.py:23:58
    movl    88(%rsp), %ecx
    imull   %edi, %ecx
    movslq  %ecx, %rcx
    .loc    1 21 27                         # addmm.py:21:27
    leaq    (%r8,%rcx,4), %rcx
    vxorps  %xmm2, %xmm2, %xmm2
    xorl    %r9d, %r9d
    xorl    %r10d, %r10d
    movq    %rbp, %rdx
    .loc    1 0 27 is_stmt 0                # :0:27
.Ltmp2:
    .p2align    4
.LBB0_10:                               #   Parent Loop BB0_8 Depth=1
                                        #     Parent Loop BB0_9 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
    .loc    1 22 56 is_stmt 1               # addmm.py:22:56
    movslq  %r9d, %r9
    .loc    1 22 32 is_stmt 0               # addmm.py:22:32
    vmovss  (%r14,%r9,4), %xmm3             # xmm3 = mem[0],zero,zero,zero
    .loc    1 23 40 is_stmt 1               # addmm.py:23:40
    movslq  %r10d, %r10
    .loc    1 24 23                         # addmm.py:24:23
    vfmadd231ss (%rcx,%r10,4), %xmm3, %xmm2 # xmm2 = (xmm3 * mem) + xmm2
    .loc    1 21 27                         # addmm.py:21:27
    addl    %r15d, %r10d
    addl    %r12d, %r9d
    decq    %rdx
    jne .LBB0_10
# %bb.11:                               # %._crit_edge.us.us.us
                                        #   in Loop: Header=BB0_9 Depth=2
    .loc    1 25 56                         # addmm.py:25:56
    movl    %r11d, %ecx
    imull   %edi, %ecx
    .loc    1 25 52 is_stmt 0               # addmm.py:25:52
    movslq  %ecx, %rcx
    .loc    1 26 51 is_stmt 1               # addmm.py:26:51
    movl    %eax, %edx
    imull   %edi, %edx
    .loc    1 26 47 is_stmt 0               # addmm.py:26:47
    movslq  %edx, %rdx
    .loc    1 27 28 is_stmt 1               # addmm.py:27:28
    vmulss  (%r13,%rcx,4), %xmm1, %xmm3
    .loc    1 27 36 is_stmt 0               # addmm.py:27:36
    vfmadd213ss %xmm3, %xmm0, %xmm2     # xmm2 = (xmm0 * xmm2) + xmm3
    .loc    1 27 21                         # addmm.py:27:21
    vmovss  %xmm2, (%rbx,%rdx,4)
    .loc    1 19 25 is_stmt 1               # addmm.py:19:25
    incq    %rdi
    cmpq    %rsi, %rdi
    jne .LBB0_9
# %bb.12:                               # %._crit_edge5.split.us.us.us
                                        #   in Loop: Header=BB0_8 Depth=1
    .loc    1 0 25 is_stmt 0                # addmm.py:0:25
    movq    -8(%rsp), %rdi                  # 8-byte Reload
    .loc    1 18 21 is_stmt 1               # addmm.py:18:21
    incq    %rdi
    cmpq    -32(%rsp), %rdi                 # 8-byte Folded Reload
    jne .LBB0_8
    jmp .LBB0_13
.LBB0_3:                                # %.lr.ph4.us.preheader
    .loc    1 0 21 is_stmt 0                # addmm.py:0:21
    vxorps  %xmm2, %xmm2, %xmm2
    vmulss  %xmm2, %xmm0, %xmm0
    xorl    %edx, %edx
    .p2align    4
.LBB0_4:                                # %.lr.ph4.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
    .loc    1 25 40 is_stmt 1               # addmm.py:25:40
    movl    96(%rsp), %ecx
    imull   %edx, %ecx
    .loc    1 25 36 is_stmt 0               # addmm.py:25:36
    movslq  %ecx, %rcx
    movq    -24(%rsp), %rdi                 # 8-byte Reload
    leaq    (%rdi,%rcx,4), %rcx
    .loc    1 26 35 is_stmt 1               # addmm.py:26:35
    movl    112(%rsp), %edi
    imull   %edx, %edi
    .loc    1 26 31 is_stmt 0               # addmm.py:26:31
    movslq  %edi, %rdi
    movq    56(%rsp), %r8
    leaq    (%r8,%rdi,4), %rdi
    xorl    %r8d, %r8d
    xorl    %r9d, %r9d
    movq    %rsi, %r10
    .loc    1 0 31                          # :0:31
.Ltmp3:
    .p2align    4
.LBB0_5:                                #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
    .loc    1 25 52 is_stmt 1               # addmm.py:25:52
    movslq  %r8d, %r8
    .loc    1 26 47                         # addmm.py:26:47
    movslq  %r9d, %r9
    vmovss  (%rcx,%r8,4), %xmm2             # xmm2 = mem[0],zero,zero,zero
    .loc    1 27 36                         # addmm.py:27:36
    vfmadd213ss %xmm0, %xmm1, %xmm2     # xmm2 = (xmm1 * xmm2) + xmm0
    .loc    1 27 21 is_stmt 0               # addmm.py:27:21
    vmovss  %xmm2, (%rdi,%r9,4)
    .loc    1 19 25 is_stmt 1               # addmm.py:19:25
    addl    %eax, %r9d
    addl    %r11d, %r8d
    decq    %r10
    jne .LBB0_5
# %bb.6:                                # %._crit_edge5.split.us6
                                        #   in Loop: Header=BB0_4 Depth=1
    .loc    1 18 21                         # addmm.py:18:21
    incq    %rdx
    cmpq    -32(%rsp), %rdx                 # 8-byte Folded Reload
    jne .LBB0_4
.LBB0_13:                               # %._crit_edge
    .loc    1 18 4 epilogue_begin           # addmm.py:18:4
    popq    %rbx
    .cfi_def_cfa_offset 48
    popq    %r12
    .cfi_def_cfa_offset 40
    popq    %r13
    .cfi_def_cfa_offset 32
    popq    %r14
    .cfi_def_cfa_offset 24
    popq    %r15
    .cfi_def_cfa_offset 16
    popq    %rbp
    .cfi_def_cfa_offset 8
    retq
.Ltmp4:
.Lfunc_end0:
    .size   addmm, .Lfunc_end0-addmm
    .cfi_endproc
                                        # -- End function
    .section    .debug_abbrev,"",@progbits
    .byte   1                               # Abbreviation Code
    .byte   17                              # DW_TAG_compile_unit
    .byte   0                               # DW_CHILDREN_no
    .byte   37                              # DW_AT_producer
    .byte   14                              # DW_FORM_strp
    .byte   19                              # DW_AT_language
    .byte   5                               # DW_FORM_data2
    .byte   3                               # DW_AT_name
    .byte   14                              # DW_FORM_strp
    .byte   16                              # DW_AT_stmt_list
    .byte   23                              # DW_FORM_sec_offset
    .byte   27                              # DW_AT_comp_dir
    .byte   14                              # DW_FORM_strp
    .byte   17                              # DW_AT_low_pc
    .byte   1                               # DW_FORM_addr
    .byte   18                              # DW_AT_high_pc
    .byte   6                               # DW_FORM_data4
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
    .byte   0                               # EOM(3)
    .section    .debug_info,"",@progbits
.Lcu_begin0:
    .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
    .short  4                               # DWARF version number
    .long   .debug_abbrev                   # Offset Into Abbrev. Section
    .byte   8                               # Address Size (in bytes)
    .byte   1                               # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
    .long   .Linfo_string0                  # DW_AT_producer
    .short  2                               # DW_AT_language
    .long   .Linfo_string1                  # DW_AT_name
    .long   .Lline_table_start0             # DW_AT_stmt_list
    .long   .Linfo_string2                  # DW_AT_comp_dir
    .quad   .Lfunc_begin0                   # DW_AT_low_pc
    .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
.Ldebug_info_end0:
    .section    .debug_str,"MS",@progbits,1
.Linfo_string0:
    .asciz  "triton"                        # string offset=0
.Linfo_string1:
    .asciz  "addmm.py"                      # string offset=7
.Linfo_string2:
    .asciz  "/home/uzairn/github/KernMLOps/python/kernmlops/compiler/extern_kernels" # string offset=16
    .section    ".note.GNU-stack","",@progbits
    .section    .debug_line,"",@progbits
.Lline_table_start0:
