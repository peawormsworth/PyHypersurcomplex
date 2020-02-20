#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
involution.albegra Unit Tests

general class wide, product table, algebraic, identity and quantum emulation tests

  file: test.py
source: https://github.com/peawormsworth/PyInvolution
author: Jeffrey B Anderson - truejeffanderson at gmail.com
"""

from involution.algebra import *
from surreal import Surreal
from surreal import creation
import unittest
import pandas as pd

VERBOSE = 1
DEBUG   = 1

# Test Data...

complex_table = """
    1  i
    i -1
"""

dual_table = """
    1  i
    i  0
"""

split_table = """
    1  i
    i  1
"""

quaternion_table = """
    1  i  j  k
    i -1  k -j
    j -k -1  i
    k  j -i -1
"""

split_quaternion_table = """
    1  i  j  k
    i -1  k -j
    j -k  1 -i
    k  j  i  1 
"""

dual_complex_table = """
    1  i  j  k
    i -1  k -j
    j -k  0  0
    k  j  0  0
"""

hyperbolic_quaternion_table = """
    1  i  j  k
    i  1  k  j
    j -k  1 -i
    k -j  i -1
"""

# not currently used...
bicomplex_table = """
    1  i  j  k
    i -1  k -j
    j  k  1  i
    k -j  i -1
"""

# not currently used...
split_quaternion_table = """
    1  i  j  k
    i -1  k -j
    j -k  1 -i
    k  j  i  1
"""

octonion_table = """
    1  i  j  k  l  m  n  o  
    i -1  k -j  m -l -o  n 
    j -k -1  i  n  o -l -m 
    k  j -i -1  o -n  m -l 
    l -m -n -o -1  i  j  k 
    m  l -o  n -i -1 -k  j
    n  o  l -m -j  k -1 -i 
    o -n  m  l -k -j  i -1
"""

split_octonion_table = """
    1  i  j  k  l  m  n  o  
    i -1  k -j -m  l -o  n 
    j -k -1  i -n  o  l -m 
    k  j -i -1 -o -n  m  l 
    l  m  n  o  1  i  j  k 
    m -l -o  n -i  1  k -j 
    n  o -l -m -j -k  1  i 
    o -n  m -l -k  j -i  1
"""

dual_quaternion_table = """
    1  i  j  k  l  m  n  o
    i -1  k -j  m -l  o -n
    j -k -1  i  n -o -l  m
    k  j -i -1  o  n -m -l
    l -m -n -o  0  0  0  0
    m  l  o -n  0  0  0  0
    n -o  l  m  0  0  0  0
    o  n -m  l  0  0  0  0
"""

sedenion_table = """
    1  i  j  k  l  m  n  o  p  q  r  s  t  u  v  w
    i -1  k -j  m -l -o  n  q -p -s  r -u  t  w -v
    j -k -1  i  n  o -l -m  r  s -p -q -v -w  t  u
    k  j -i -1  o -n  m -l  s -r  q -p -w  v -u  t
    l -m -n -o -1  i  j  k  t  u  v  w -p -q -r -s
    m  l -o  n -i -1 -k  j  u -t  w -v  q -p  s -r
    n  o  l -m -j  k -1 -i  v -w -t  u  r -s -p  q
    o -n  m  l -k -j  i -1  w  v -u -t  s  r -q -p
    p -q -r -s -t -u -v -w -1  i  j  k  l  m  n  o
    q  p -s  r -u  t  w -v -i -1 -k  j -m  l  o -n
    r  s  p -q -v -w  t  u -j  k -1 -i -n -o  l  m
    s -r  q  p -w  v -u  t -k -j  i -1 -o  n -m  l
    t  u  v  w  p -q -r -s -l  m  n  o -1 -i -j -k
    u -t  w -v  q  p  s -r -m -l  o -n  i -1  k -j
    v -w -t  u  r -s  p  q -n -o -l  m  j -k -1  i
    w  v -u -t  s  r -q  p -o  n -m -l  k  j -i -1
"""

# Class tools...

def unit_list (obj):
    d = 2 ** len(obj.dp)
    return [ obj( ( [0]*i + [1] + [0]*(d-i-1) )) for i in range(d) ]

def generate_str (obj):
    """create a multiplication table for a given Algebra object and return the elements in string format"""
    d = obj.dim(obj)
    s = creation(days=7)
    zer = s[0]
    one = s[1]

    units = [ obj( ( [zer]*i + [one] + [zer]*(d-i-1) )) for i in range(d) ]
    table = []
    raw_table = []
    for j in units:
        table.append([])
        raw_table.append([])
        for i in units:
            if DEBUG: raw_table[-1].append(str(j*i))
            table[-1].append(str(obj([c.name_in(s) for c in (j*i).state])))
            if DEBUG: print('{} × {} = {}'.format(j,i,j*i))
    return table
        
# Verbose output formats

def _test_unit_multiplication (self,expect,calc):
    "generic unit product table"
    imaginaries = '1ijklmnopqrstuvw'
    n  = self.obj.dim(self.obj)
    il = list(imaginaries[:n])

    if DEBUG: print("\ncalc:   {0}\nexpect: {1}\nil: {2}".format(calc, expect, il))

    if VERBOSE: 
        print(_verbose_unit_multiplication().format(
            object           = self.obj.__name__,
            expected_table   = pd.DataFrame( expect, index = il, columns = il ),
            calculated_table = pd.DataFrame(   calc, index = il, columns = il )
        ))
    self.assertListEqual(calc, expect)


def _verbose_unit_multiplication ():
   return """

=== expected {object} multiplication table ===

{expected_table}

=== calculated {object} multiplication table ===

{calculated_table}

...
"""

# Classes

class TestComplex(unittest.TestCase):
    obj = Complex

    def test_unit_multiplication (self):
        "Complex number unit product table"
        expect = [ a.split() for a in complex_table.strip().split("\n") ]
        print('expect:',expect)
        calc = generate_str(self.obj)
        _test_unit_multiplication(self, expect=expect, calc=calc)

class TestDual(unittest.TestCase):
    obj = Dual

    def test_unit_multiplication (self):
        "Dual number unit product table"
        expect = [ a.split() for a in dual_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self, expect=expect, calc=calc)

class TestOctonion(unittest.TestCase):
    obj = Octonion

    def test_unit_multiplication (self):
        "Octonion unit product table"
        expect = [ a.split() for a in octonion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)

class TestSedenion(unittest.TestCase):
    obj = Sedenion

    def test_unit_multiplication (self):
        "Sedenion unit product table"
        expect = [ a.split() for a in sedenion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)


# Exotic Algebra tests

class TestSplitQuaternion(unittest.TestCase):
    obj = SplitQuaternion

    def test_unit_multiplication (self):
        "Split Quaternion unit product table"
        expect = [ a.split() for a in split_quaternion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)

class TestSplitOctonion(unittest.TestCase):
    obj = SplitOctonion

    def test_unit_multiplication (self):
        "SplitOctonion unit product table"
        expect = [ a.split() for a in split_octonion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)

class TestDualComplex(unittest.TestCase):
    obj = DualComplex

    def test_unit_multiplication (self):
        "DualComplex unit product table"
        expect = [ a.split() for a in dual_complex_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)


class TestDualQuaternion(unittest.TestCase):
    obj = DualQuaternion

    def test_unit_multiplication (self):
        "DualQuaternion unit product table"
        expect = [ a.split() for a in dual_quaternion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)


class TestHyperbolicQuaternion(unittest.TestCase):
    obj = HyperbolicQuaternion

    def test_unit_multiplication (self):
        "HyperbolicQuaternion unit product table"
        expect = [ a.split() for a in hyperbolic_quaternion_table.strip().split("\n") ]
        calc = generate_str(self.obj)
        _test_unit_multiplication(self,expect=expect,calc=calc)


# run all tests on execute...

if __name__ == '__main__':

    unittest.main(verbosity=2)

