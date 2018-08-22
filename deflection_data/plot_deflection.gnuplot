set style line 1 lt 1 lc 3 lw 3 pt 1
set term pngcairo enhanced font "Verdana, 10"
set output "deflection_settings.png"

set title "Deflection Angle Stage Settings - 185 A"
set ylabel "Beam Energy [GeV]
set xlabel "Stage Position Setting"

plot "deflection_data.txt" using 2:1 ls 1 notitle 
