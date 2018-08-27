set style line 11 lc rgb "#000000' lt 1
set border 3 back ls 11

set style line 12 lc rgb "#808080" lt 0 lw 1
set grid back ls 12

set term pngcairo enhanced font "Times New Roman, 12"
set output "deflection_settings.png"

set title "Deflection Position - 185 A"
set xlabel "Beam Energy [GeV]
set ylabel "Deflection X position [mm] "

set xtics nomirror
set ytics nomirror

set xtics (1.4, 2.0, 2.4, 3.0, 3.4, 4.0, 4.4, 5)

set xrange[1:5.4]
plot 'deflection_data.txt' using 1:2:4 with yerrorbars lc 3 lw 2 ps 2 notitle
