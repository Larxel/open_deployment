# Open Deployment
Open deployment is a simple, easy to use package that allows you to prepare all the files need to deploy your machine learning models as a function or container with ease.

Open deployment focuses on making ml deployment easy, so that you can focus on the cooler stuff instead. ˙ ͜ʟ˙

# Examples
```python
import open_deployment as od

od.deploy_faas(version='1.0.0',
               load_ml_model_function = load_model,
               prediction_function    = predict,
               model_file = 'model.pkl',
               preprocessing_function = feature_engineering
              )
```

For a complete example, please have a look at the `examples` folder.

# Main Features
- Generates zip file for seamless FaaS deployment
- Flask RestPlus API (REST and with automatically generated swagger documentation)
- Handy Health checks
- Auto split code into components (preprocessing_functions.py, main.py)

# Future Features
- AWS lambda support
- Unit testing for APIs
- Roleplaying as DataOps ˙ ͜ʟ˙

# Installation
```sh
$ pip install open_deployment
```

# License
MIT, yeet!
