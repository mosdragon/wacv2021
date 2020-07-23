# Labelme Dataset
[Labelme][labelme] is a tool for image annotations. It allows users to annotate
images one-by-one with polygons for each instance of a class label.

Here, we've provided a modified version of the [script][conversion_script] to
allow you to convert a labelme dataset into a proper COCO-formatted dataset for
use with any Segmentation model that supports the COCO format.

We also provide you with the commands needed to run labelme with the same labels
we've used to make your own annotations, and then the conversion script to
create the final COCO-formatted dataset.

## Setup
__Note:__ To use `labelme` to make your own annotations, you will need to be on
a machine with a graphical interface (Windows, MacOS, or a Linux machine with a
graphical interface).

```bash
pip install labelme
```

## Create Annotations
__Note:__ You can skip this step if you don't want to annotate images yourself.

Following the example of [segmenting a single image][single_image], we've
created the following bash script to run `labelme` on each of the images in
`train_images` and `val_images` and annotate them. With each Labelme window,
you'll draw polygons to mark the cat's ears one-by-one, and click "Save" to
move onto the next image.

This is a bash script, so it may need to be modified slightly for use on Windows:
```bash
# Run from the raw_labelme directory.
cd raw_labelme

# Iterate over all training images first.
for img_path in $(find train_images | grep jpg); do;

# Save annotations for train_images/img_x.jpg in train_annotations/img_x.json.
basename=$(basename ${img_path/jpg/json})
annot_path="train_annotations/img_${basename}"

# Call labelme with the paths to the image and annotation files.
labelme  --labels labels.txt --keep-prev --nodata $img_path -O $annot_path
done

# Repeat the same for validation.
for img_path in $(find val_images | grep jpg); do;
basename=$(basename ${img_path/jpg/json})
annot_path="val_annotations/img_${basename}"
labelme  --labels labels.txt --keep-prev --nodata $img_path -O $annot_path
done
```

## Export to COCO Format
Finally, once we have our raw polygon annotations, we can export them all to
the COCO format using this `labelme2coco.py` script, slightly modified from the
[original][conversion_script].

```
python labelme2coco.py ./raw_labelme
```

This will generate a new folder called `coco_labelme` in the current directory.
The contents of the directory will be a dataset formatted exactly like COCO
2014.

---
[labelme]: https://github.com/wkentaro/labelme
[conversion_script]: https://github.com/wkentaro/labelme/blob/master/examples/semantic_segmentation/labelme2voc.py
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
