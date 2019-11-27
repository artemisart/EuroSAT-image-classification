# EuroSAT image classification

```sh
wget -P data --continue http://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip data/EuroSAT.zip -d data -q
```

I decided to use Pytorch as it seemed appropriate and I had more experience with
this framework.  For loading and handling the dataset I used an `ImageLoader` to
integrate nicely with pytorch pipelines (e.g. multithreaded data loaders).
