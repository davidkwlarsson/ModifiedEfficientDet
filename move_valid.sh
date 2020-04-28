
BASE="/home/sofie.hellmark/FreiHAND_pub_v2/"
#BASE="/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
FILES=$BASE
FILES+="training/rgb/*"
echo "${FILES}"	 	


DIRv=$BASE
DIRv+="validation"
DIRt=$BASE
DIRt+="test"
echo "${DIRv}"	 	

mkdir $DIRv
mkdir $DIRt

DIRv+="/rgb"
DIRt+="/rgb"
echo "${DIRv}"	 	

mkdir $DIRv
mkdir $DIRt

total_samples=130240
train_samples=$((total_samples/20*17))
validation_samples=$((total_samples/10))
test_samples=$((total_samples/20))
val_part=$((validation_samples/4))
test_part=$((test_samples/4))
train_part=$((train_samples/4))
total_part=$((total_samples/4))


i=0
echo "i = $i"
for f in $FILES
do
        if [ "$i" -ge "$train_part" ] && [ "$i" -lt "$((train_part + val_part))" ]; then mv $f $DIRv;
        elif [ "$i" -ge "$((train_part + val_part))" ] && [ "$i" -lt "$((train_part + val_part + test_part))" ]; then mv $f $DIRt;
        elif [ "$i" -ge "$((total_part + train_part))" ] && [ "$i" -lt "$((total_part + train_part + val_part))" ]; then mv $f $DIRv;
        elif [ "$i" -ge "$((total_part + train_part + val_part))" ] && [ "$i" -lt "$((total_part + train_part + val_part + test_part))" ]; then mv $f $DIRt;
        elif [ "$i" -ge "$((total_part * 2 + train_part))" ] && [ "$i" -lt "$((total_part * 2 + train_part + val_part))" ]; then mv $f $DIRv;
        elif [ "$i" -ge "$((total_part * 2 + train_part + val_part))" ] && [ "$i" -lt "$((total_part * 2 + train_part + val_part + test_part))" ]; then mv $f $DIRt;
        elif [ "$i" -ge "$((total_part * 3 + train_part))" ] && [ "$i" -lt "$((total_part * 3 + train_part + val_part))" ]; then mv $f $DIRv;
        elif [ "$i" -ge "$((total_part * 3 + train_part + val_part))" ] && [ "$i" -lt "$((total_part * 3 + train_part + val_part + test_part))" ]; then mv $f $DIRt;fi; 
        let "i += 1"
done
