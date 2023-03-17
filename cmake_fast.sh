cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DGALAX_LINK_OMP=ON -DCMAKE_CXX_FLAGS="-mavx2" ..
cmake --build "build" --config Release --parallel
./build/bin/galax -c CPU_FAST
