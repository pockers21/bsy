rm -rf CMakeFiles
rm -rf CMakeCache.txt
cmake .
make
./tools/bsy.bin
