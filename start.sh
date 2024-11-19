rm -rf build
python3 setup.py build
rm *.so
mv build/lib.*/*.so .
python3 tests/test.py
