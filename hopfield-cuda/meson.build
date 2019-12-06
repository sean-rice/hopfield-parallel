project('hopfield-cuda', 'c', 'cuda')

cxx = meson.get_compiler('c')
if cxx.get_id() in ['gcc', 'clang']
  add_global_arguments('-Wall', language : 'c')
  add_global_arguments('-Wall', language : 'cuda')
elif cxx.get_id() == 'intel'
  add_global_arguments('-w2', language : 'c')
endif


sources = ['src/hopfield_network.c', 'src/hopfield_pattern.c', 'src/main.c', 'src/util.c']

cpu_sources = ['src/hopfield_retrieve.c']
cuda_sources = ['src/hopfiled_retriece.cu']

includes = include_directories('include')


executable('main', sources + cpu_sources, include_directories : includes)
executable('maincuda', sources + cuda_sources, include_directories : includes)
