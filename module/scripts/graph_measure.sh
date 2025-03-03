# Goal is to graph keeping data_size at 256
#./print_measure.sh $1 | tail -n +2 |awk '$2 == 256 {print $3, $4/$1}'

alias cwk='awk -F, -v OFS=","'

get_data() {
    ./scripts/print_measure.sh $1 | tail -n +2 | cwk '{print $2, $3, $4/$1, sqrt($5)/$1}'
}

KMOD_ARRAY=$(get_data kmod-array)
KMOD_MAP=$(get_data kmod-map)
EBPF_MAP=$(get_data ebpf-map)
USER_MMAP=$(get_data user-mmap)
USER_KSCHED=$(get_data user-ksched)

echo kmod-array ebpf-map kmod-map user-mmap user-ksched
for i in 8 16 32 64 128 256 512; do
    printf "$(printf "${KMOD_ARRAY}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $3}')"
    printf "\t$(printf "${EBPF_MAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $3}')"
    printf "\t$(printf "${KMOD_MAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $3}')"
    printf "\t$(printf "${USER_MMAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $3}')"
    printf "\t$(printf "${USER_KSCHED}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $3}')"
    printf "\t$i"
    printf "\t$(printf "${KMOD_ARRAY}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $4}')"
    printf "\t$(printf "${EBPF_MAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $4}')"
    printf "\t$(printf "${KMOD_MAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $4}')"
    printf "\t$(printf "${USER_MMAP}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $4}')"
    printf "\t$(printf "${USER_KSCHED}" | cwk -v DS=$i '$2 == 256 && $1 == DS {print $4}')"
    printf "\n"
done

#echo kmod-array ebpf-map kmod-map user-mmap user-ksched
#for i in 32 64 128 256 512 1024 2048; do
#  printf "$(printf "${KMOD_ARRAY}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $3}')"
#  printf "\t$(printf "${EBPF_MAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $3}')"
#  printf "\t$(printf "${KMOD_MAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $3}')"
#  printf "\t$(printf "${USER_MMAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $3}')"
#  printf "\t$(printf "${USER_KSCHED}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $3}')"
#  printf "\t$i"
#  printf "\t$(printf "${KMOD_ARRAY}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $4}')"
#  printf "\t$(printf "${EBPF_MAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $4}')"
#  printf "\t$(printf "${KMOD_MAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $4}')"
#  printf "\t$(printf "${USER_MMAP}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $4}')"
#  printf "\t$(printf "${USER_KSCHED}" | cwk -v MS=$i '$2 == MS && $1 == 256 {print $4}')"
#  printf "\n"
#done
