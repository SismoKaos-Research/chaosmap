project(
  'chaosmap',
  'cpp',
  meson_version : '>= 1.3.0',
  version : '0.2.0',
  default_options : [
    'cpp_std=c++23',
    'warning_level=3',
    'werror=true',
    'optimization=3',
    'b_ndebug=if-release',
  ],
)

threads_dep = dependency('threads')
libzip_dep  = dependency('libzip')
toml_dep    = dependency('tomlplusplus')
fftw3_dep   = dependency('fftw3')
tbb_dep     = dependency('tbb')
lapack_dep  = dependency('lapack')

common_deps = [
  threads_dep,
  libzip_dep,
  toml_dep,
  fftw3_dep,
  tbb_dep,
  lapack_dep,
]

inc = include_directories('src')

exe = executable(
  'chaosmap',
  ['app/chaosmap.cpp'],
  include_directories : inc,
  dependencies : common_deps,
  install : true,
)

test_exe = executable(
  'test_chaosmap',
  ['tests/test_main.cpp'],
  include_directories : inc,
  dependencies : common_deps,
)

test('unit_and_integration', test_exe, workdir : meson.project_source_root())
