 #set term wxt
 set term postscript landscape monochrome solid "Helvetica" 22
 set output "./OUTPUT_FILES/gridfile.ps"
 #set xrange [  0.0000000E+00 :   179.0000     ]
 #set yrange [   1.000000     :   60.00000     ]
 set size ratio -1
 set loadpath "./OUTPUT_FILES/"
 plot "gridfile.gnu" title "Macrobloc mesh" w l
 pause -1 "Hit any key..."
