#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:58:55 2017

@author: hhjiang
"""
import parse
import ID3
import node
import unit_tests


data = parse.parse('/Users/hhjiang/Documents/pystuff/EECS349/PS2/house_votes_84.data')
xr,acc1,acc2 = ID3.vary_test(data,10,300,100)
ID3.plot_results(xr,acc1,acc2)
N = ID3.ID3(data,0)
N.print_tree()




