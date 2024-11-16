cmake -S . -B build
cmake --build build
mv build/*.so op
python3 test.py
