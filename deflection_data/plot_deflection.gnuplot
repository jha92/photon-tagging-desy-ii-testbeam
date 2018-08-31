set style line 11 lc rgb "#000000" lt 1
set border 3 back ls 11

set style line 12 lc rgb "#808080" lt 0 lw 1
set grid back ls 12

set term png size 1100, 600 enhanced font "Times New Roman, 20"
set output "deflection_settings.png"

set title "Deflection Position in 0.18 T Magnetic Field"
set xlabel "Beam Energy [GeV]"
set ylabel "Deflection X position [mm] "

set xtics nomirror
set ytics nomirror

set xtics (1.4, 2.0, 2.4, 3.0, 3.4, 4.0, 4.4, 5)

set xrange[1:4.8]
set yrange[0:25]
plot 'deflection_data.txt' using 1:2:4 with yerrorbars lc rgb "#1E90FF" lw 2 ps 4 title "Measurement" ,\
     'deflection_data.txt' using 1:5:6 with yerrorbars lc rgb "#800080" lw 2 ps 4 title "SLIC Simulation" 
