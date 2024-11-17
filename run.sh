rm -rf build
rm op/*.so
cmake -S . -B build
cmake --build build
mv build/*.so op
python3 test.py
