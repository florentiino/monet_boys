# Using machine learning to create drawings in the style of famous painters

# Data analysis
- Description: We analyzed various image datasets of painters to predict brushstrokes and train a model to transform a given photo.
- Data Source: [Model Zoo](https://github.com/junyanz/CycleGAN#model-zoo) monet2photo, vangogh2photo, ukiyoe2photo, cezanne2photo: The art images were downloaded from [Wikiart](https://www.wikiart.org/). The real photos are downloaded from Flickr using the combination of the tags landscape and landscapephotography. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
- Type of analysis: Implementing [cycle GAN](https://software.intel.com/content/www/us/en/develop/articles/improving-cycle-gan-using-intel-ai-devcloud.html) to generate the most accurate painting possible 
- Functional frontend can be found [here](https://showmeyourgans.herokuapp.com/) 


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for monet_boys in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/monet_boys`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "monet_boys"
git remote add origin git@github.com:{group}/monet_boys.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
monet_boys-run
```

# Install

Go to `https://github.com/{group}/monet_boys` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/monet_boys.git
cd monet_boys
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
monet_boys-run
```
