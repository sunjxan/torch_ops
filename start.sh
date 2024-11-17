rm -rf build
cmake -S . -B build
cmake --build build
rm op/*.so
mv build/*.so op
python3 test.py
