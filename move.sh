#FILES=/Users/Sofie/exjobb/ModifiedEfficientDet/test/*
FILES=/home/sofie.hellmark/FreiHAND_pub_v2/training/rgb/*
#$train_range_1_top = 10#32000
#$train_range_2_top = 20#64560
#$train_range_2_bot = 15#32560
#$train_range_3_bot = 25#65120
#$train_range_3_top = 30#97120
#$train_range_4_bot = 35#97680
#$train_range_4_top = 40#129680
i=0
echo "i = $i"
for f in $FILES
do
        if [ "$i" -ge 32000 ] && [ "$i" -le 32559 ]; then mv $f /home/sofie.hellmark/FreiHAND_pub_v2/training/rgb/validation;
        elif [ "$i" -ge 64560 ] && [ "$i" -le 65119 ]; then mv $f /home/sofie.hellmark/FreiHAND_pub_v2/training/rgb/validation;
        elif [ "$i" -ge 97120 ] && [ "$i" -le 97679 ]; then mv $f /home/sofie.hellmark/FreiHAND_pub_v2/training/rgb/validation;
        elif [ "$i" -ge 129680 ]; then mv $f /home/sofie.hellmark/FreiHAND_pub_v2/training/rgb/validation;fi

        let "i += 1"
echo $f
done
