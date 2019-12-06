project('hopfield-cuda', 'c')

sources = [
  'src/hopfield_network.c',
  'src/hopfield_pattern.c',
  'src/main.c',
  'src/util.c'
]
cpu_sources = ['src/hopfield_retrieve.c']
cuda_sources = ['src/hopfield_retrieve.cu']
includes = include_directories('include')

cc = meson.get_compiler('c')
if cc.get_id() in ['gcc', 'clang']
  add_global_arguments('-Wall', language : 'c')
elif cc.get_id() == 'intel'
  add_global_arguments('-w2', language : 'c')
endif
m_dep = cc.find_library('m', required : true)

executable('main', sources + cpu_sources, include_directories : includes, dependencies: m_dep)

if add_languages('cuda', required : false)
  executable('maincuda', sources + cuda_sources, include_directories : includes, dependencies: m_dep)
endif
