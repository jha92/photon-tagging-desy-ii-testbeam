set term pngcairo size 1100,600 enhanced font "Times New Roman, 20"
set output "photon_rates.png"

set style line 11 lc rgb "#000000" lt 1
set border 3 back ls 11

set style line 12 lc rgb "#808080" lt 0 lw 1
set grid back ls 12

set xtics nomirror
set ytics nomirror

set style line 1 lt 1 lc rgb "#1E90FF" lw 2 ps 5
set style line 2 lt 1 lc rgb "#FFD700" lw 2 ps 5

set title "Photon production rate 3mm Copper Target"
set xlabel "Photon Energy [GeV]"
set ylabel "Count Rate [1/s]"
set xrange [0.6:4.0]
set xtics (1.0, 1.6, 2.0, 2.6, 3.0, 3.6)
#set multiplot layout 2,1 rowsfirst
# plot 1
#set key outside
set logscale y 
plot "photon_rates.txt" using 3:($4/10):($5/10) w yerrorbars ls 1 title "2 Coincidence",\
     "photon_rates.txt" using 3:($6/10):($7/10) w yerrorbars ls 2 title "3 Coincidence"

#plot 2
#set title "Coincidence Ratio"
#set xlabel "Beam Energy [GeV]
#set ylabel "Coincidence Ratio"
#plot "photon_rates.txt" using 2:($4/$3) ls 3 title "3mm", \
#	 "photon_rates.txt" using 2:($6/$5) ls 4 title "4.5mm"
#unset multiplot
