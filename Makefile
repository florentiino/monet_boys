# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* monet_boys/*.py

black:
	@black scripts/* monet_boys/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr monet_boys-*.dist-info
	@rm -fr monet_boys.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GCP SETUP
# ----------------------------------

# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
LOCAL_PATH= '../raw_data/test/test_monet.jpg'

# project id
PROJECT_ID='batch-672-gan-monet'

# bucket name
BUCKET_NAME='bucket-monet-gan'

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER='raw_data/test'

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=europe-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

run_better_api:
	uvicorn api.fast:app --reload
