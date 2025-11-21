# Basic examples
add_executable(quadrotor_tracking quadrotor_tracking.cpp)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympcstatic)

add_executable(quadrotor_hovering quadrotor_hovering.cpp)
target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympcstatic)

add_executable(cartpole_example cartpole_example.cpp)
target_link_libraries(cartpole_example LINK_PUBLIC tinympcstatic)

# Linear constraint example
add_executable(quadrotor_linear_constraints quadrotor_linear_constraints.cpp)
target_link_libraries(quadrotor_linear_constraints LINK_PUBLIC tinympcstatic)

# Time-varying Linear constraint example
add_executable(quadrotor_tv_linear_constraints quadrotor_tv_linear_constraints.cpp)
target_link_libraries(quadrotor_tv_linear_constraints LINK_PUBLIC tinympcstatic)

# Conic example
add_executable(rocket_landing_mpc rocket_landing_mpc.cpp)
target_link_libraries(rocket_landing_mpc LINK_PUBLIC tinympcstatic)

# PSD demo example
add_executable(tiny_psd_demo tiny_psd_demo.cpp)
target_link_libraries(tiny_psd_demo LINK_PUBLIC tinympcstatic)

# PSD -> TV pipeline demo
add_executable(tiny_psd_tv_demo tiny_psd_tv_demo.cpp)
target_link_libraries(tiny_psd_tv_demo LINK_PUBLIC tinympcstatic)

# PSD-regularized TV demo
add_executable(tiny_psd_reg_tv_demo tiny_psd_reg_tv_demo.cpp)
target_link_libraries(tiny_psd_reg_tv_demo LINK_PUBLIC tinympcstatic)

# PSD -> TV pipeline narrow-2D demo
add_executable(tiny_psd_tv_narrow2d tiny_psd_tv_narrow2d.cpp)
target_link_libraries(tiny_psd_tv_narrow2d LINK_PUBLIC tinympcstatic)

# PSD + TV narrow-2D linear corridor demo
add_executable(tiny_tv_psd_reg_narrow2d tiny_tv_psd_reg_narrow2d.cpp)
target_link_libraries(tiny_tv_psd_reg_narrow2d LINK_PUBLIC tinympcstatic)

# U-shape demos (PSD vs TV)
add_executable(tiny_psd_ushape_demo tiny_psd_ushape_demo.cpp)
target_link_libraries(tiny_psd_ushape_demo LINK_PUBLIC tinympcstatic)
add_executable(tiny_tv_ushape_demo tiny_tv_ushape_demo.cpp)
target_link_libraries(tiny_tv_ushape_demo LINK_PUBLIC tinympcstatic)
add_executable(tiny_cbf_ushape_demo tiny_cbf_ushape_demo.cpp)
target_link_libraries(tiny_cbf_ushape_demo LINK_PUBLIC tinympcstatic)
add_executable(tiny_psd_dynamic_demo tiny_psd_dynamic_demo.cpp)
target_link_libraries(tiny_psd_dynamic_demo LINK_PUBLIC tinympcstatic)
add_executable(tiny_tv_dynamic_demo tiny_tv_dynamic_demo.cpp)
target_link_libraries(tiny_tv_dynamic_demo LINK_PUBLIC tinympcstatic)
add_executable(tiny_cbf_dynamic_demo tiny_cbf_dynamic_demo.cpp)
target_link_libraries(tiny_cbf_dynamic_demo LINK_PUBLIC tinympcstatic)

# PSD narrow-2D corridor demo
add_executable(tiny_psd_demo_narrow2d tiny_psd_demo_narrow2d.cpp)
target_link_libraries(tiny_psd_demo_narrow2d LINK_PUBLIC tinympcstatic)

# TV-linear only demo (no PSD)
add_executable(tiny_tv_linear_demo tiny_tv_linear_demo.cpp)
target_link_libraries(tiny_tv_linear_demo LINK_PUBLIC tinympcstatic)

# TV-linear narrow-2D corridor demo
add_executable(tiny_tv_linear_demo_narrow2d tiny_tv_linear_demo_narrow2d.cpp)
target_link_libraries(tiny_tv_linear_demo_narrow2d LINK_PUBLIC tinympcstatic)

# Codegen examples
add_executable(codegen_random codegen_random.cpp)
target_link_libraries(codegen_random LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_random LINK_PUBLIC "stdc++fs")
endif(MINGW)

add_executable(codegen_cartpole codegen_cartpole.cpp)
target_link_libraries(codegen_cartpole LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_cartpole LINK_PUBLIC "stdc++fs")
endif(MINGW)
