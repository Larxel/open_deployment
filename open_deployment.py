def get_python_version(latest = False):
    '''
    Retrieve Python Version Function
    
    # Description
        Retrieves the current python version and returns it as a dockerfile formatted string

    # Arguments
        latest: bool
            Wether or not to add latest at the end of string


    # Examples
        [In]>>> import open_deployment as od
        [In]>>> od.get_python_version(latest=True)
        [Out]'python 3.7:latest'

        --------------------------------------------------------

        [In]>>> import open_deployment as od
        [In]>>> od.get_python_version(latest=False)
        [Out]'python 3.7.1'
    '''
    import sys
    python_ver = sys.version_info
    

    if latest == True:
        python_ver = 'python {}.{}'.format(python_ver[0],python_ver[1])
        python_ver = python_ver + ':latest'
    else:
        python_ver = 'python {}.{}.{}'.format(python_ver[0],python_ver[1],python_ver[2])

    return(python_ver)



def generate_freeze(filename = 'requirements.txt', extra_packages = None):
    '''
    # Freeze generator function
    
    # Description
        Executes pip freeze from within python and exports it to a text file

    # Arguments
        filename: string
            Name of the file where the python packages will be written in.
        extra_packages: list of strings
            Extra packages to be added to the file, if not present before.

    # Examples

    '''
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze

    # Retrieve requirements
    requeriments = freeze.freeze()
    requeriments = list(requeriments)

    # Add extra packages
    if extra_packages:
        for package in extra_packages:
            if package not in requeriments:
                teste.append('abcd')
                requeriments.append(package)

    # Write file
    with open(filename, 'w') as file:
        for item in requeriments:
            file.write(item + '\n')
        file.close()

    print('Dependencies successfully written in file {}!'.format(filename))



def compose_dockerfile(filename='dockerfile', os='centos:7.4', latest=True, dockerfile_content='default'):
    '''
    # Compose Dockerfile Function
    
    # Description
        Creates a dockerfile using current Python version

    # Arguments
        filename: string
            Name of the file that where the dockerfile content will be written in.
            Note that no suffix is added.
        os: string
            Name of the operational system and it's corresponding version
        latest: bool
            Wether or not to include 'latest' at the end of the os string
            This causes docker to search for the latest version within the version boundary provided by the os argument

    # Examples
        [Input]>>>  import docker_function as docker
        [Input]>>>  docker.compose_dockerfile('microservice_dockerfile', os='centos:7.3', latest=False)
        [Output]>>> Dockerfile microservice_dockerfile generated successfully!

        Generated file contents:
            FROM centos:7.3

            RUN yum update
            RUN pip install -r python_requirements.txt
            CMD ["gunicorn", "-w 4", "main:app"]

    '''
    import re

    # OS and Python versions
    if latest == True:
        os_header   = 'FROM {}:latest'.format(str(os))
    elif latest == False:
        os_header   = 'FROM {}'.format(str(os))
    else:
        raise ValueError("Invalid value for 'latest' argument")

    # Assign proper package manager depending on Linux distribution
    if re.findall('ubuntu', os_header):
        os_update = 'RUN apt-get update'
    elif re.findall('centos', os_header):
        os_update = 'RUN yum update'
    elif re.findall('mint', os_header):
        os_update = 'RUN apt-get update'
    elif re.findall('alpine', os_header):
        os_update = 'RUN apk update'
    else:
        raise ValueError('Unable to identify OS and corresponding package manager')

    # Get python version
    try:
        python  = get_python_version()
    except:
        raise ValueError('Unable to retrieve python version')

    # Python dependencies and gunicorn execution
    install_dependencies = 'RUN pip install -r requirements.txt'
    gunicorn_exec        = 'CMD ["gunicorn", "-w 4", "main:app"]'

    # Wrap it all up
    content = [os_header, '\n' * 2, os_update, '\n', install_dependencies, '\n' * 2, gunicorn_exec]

    # Write dockerfile
    with open(filename, 'w') as file:
        for piece in content:
            file.write(piece)
        file.close()

    print('Dockerfile {} generated successfully!'.format(filename))



def deploy_as_container(output_type='default',**kwargs):
    '''
    # Deploy as Container Function
    
    # Description
    Wrapper to generate python requirements file, dockerfile and API modules, all required to deploy as a container

    # Arguments
        Spoon

    # Examples
    '''

    # Generate requirements, dockerfile and generate api
    generate_freeze(**kwargs)
    compose_dockerfile()
    compose_api()

    # Assign output type to desired execution
    if output_type == 'default':
        pass
    elif output_type == 'zip':
        generate_zip()
    else:
        raise ValueError("Invalid type for 'output_type' argument")


    print('yeet!')



def generate_zip(pattern = 'application/*', local_directory_files = ['requirements.txt', 'dockerfile']):
    '''
    Generate Zip File Function

    From a fixed set of files, creates a zip file, ideal for deploying in serverless functions such as Google Cloud Functions, Amazon Lambda, and Azure Functions

    # Arguments
        None

    # Examples

        generate_zip()

        yields:
            application.zip file, containing:
                application
                    main.py
                    health_functions.py
                    preprocessing_functions.py
                dockerfile
                requirements.txt
    '''
    import glob
    import zipfile

    # Define files to be saved
    application_files = glob.glob(pattern)

    # Create zipfile
    zip_file = zipfile.ZipFile('application.zip', 'w')

    # Write files to zip
    for file in application_files:
        zip_file.write(file)

    for file in local_directory_files:
        zip_file.write(file)


    # Close file, print results
    zip_file.close()
    print('Zip file successfully generated!')
        



def compose_api(load_ml_model_function = '', prediction_function = '', directory='application', api_model='flask-restplus', version="1.0.0", title="API Template", description = "API Template for serving machine learning models", debug=True):
    '''
    # Compose API Function - Open Deployment
    
    # Description
    Creates an API from the given code, APIs can be created from both script files as well as jupyter notebooks.
    
    # Arguments
        load_ml_model_function: function
            Function that loads the model from a file into memory
        prediction_function: function
            Function that outputs a prediction, it should also include
        directory: string
            Folder where the api files will be saved on.
        api_model: string
            Type of API model that will be generated, by default the API is modeled with flask-restplus
        version: string
            String of model version - useful for controlling model performance and display at the base url of the API
        title: string
            Title of the API, display at the base url of the API
        description: string
            The description of the API that will appear close to the title, display at the base url of the API
        debug: boolean
            Wether or not to activate debug mode in the API
            Defaults to True, but in production environments it is recommended to be set as False
    # Examples

    '''
    import os
    import re
    import sys


    # Add quotes to strings that will be used as arguments when run
    version     = "'" + str(version) + "'"
    title       = "'" + str(title) + "'"
    description = "'" + str(description) + "'"

    # Select from a library of templates
    if api_model == 'flask-restplus':
        main_content = "import gc\nimport json\nfrom   flask          import Flask, request\nfrom   flask_restplus import Resource, Api, Namespace\nimport health_functions        as health\nimport preprocessing_functions as preprocessing\n\n# Aplication initialization\napp = Flask(__name__)\napi = Api(app, version={0}, title = {1}, description = {2})\n\n\n# Import configuration parameters\nwith open('app_config.json') as config_file:\n\tconfig = json.load(config_file)\n\n\n# Singleton for model deployment\nclass Singleton(object):\n    def __new__(cls, *args, **kwargs):\n        if not hasattr(cls, '_instance'):\n            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)\n        return cls._instance\n\n\n# Assign model to the singleton\nsingle_model    = Singleton()\nsingle_model.ai = {3}\n\n\n# General Health Check\nclass Health(Resource):\n\t'''\n\tGeneral Health Check\n\tReturns operational status of application and it's components\n\n\tInput: None\n\tProcess: Checks the health of every component, usually by dummy queries\n\tOutput: Status 'UP' if accessible, or else http error code\n\t'''\n\tdef get(self):\n\t\tstatus = dict()\n\t\tstatus['Application'] = 'UP'\n\t\t#status['PostgreSQL'] = health.psql_health(config)\n\t\treturn status\n\n\n\n# Core Function: Call Analysis\nclass Predict(Resource):\n\t'''\n\tInput: Some id\n\n\tProcess: Receives a Call ID, retrieves the data associated with that call from PostgreSQL.\n\tPrepares the data and predicts using viceroy lgb.\n\n\tOutput: Returns call id, wether the call is considered a lead or not, trade in tag, financing tag, banks mentioned and execution log details.\n\t'''\n\tdef get(self, some_id):\n\t\t{4}\n\n\t\tresults = dict()\n\t\tresults['id'] \t\t\t = str(some_id)\n\t\tresults['prediction'] \t = 0.99\n\t\tresults['model_version'] = {0}\n\t\treturn(results)\n\n\n\n# Assing resources and corresponding routes\napi.add_resource(Health,  '/health')\napi.add_resource(Predict, '/predict/<string:some_id>')\n\n\n# Run application\ngc.collect()\nif __name__ == '__main__':\n\tapp.run(host = '0.0.0.0', debug = {5})\n".format(version, title, description, load_ml_model_function, prediction_function, debug)
    else:
        raise ValueError('Only flask-restplus is supported right now')


    # Create API folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print('Directory {} already exists'.format(directory))


    # Identify wether the current file is a notebook or a script
    filename, filetype   = get_current_filename()
    current_file_content = open(filename,'r').read()

    # Scan file for Preprocessing content
    if filetype == 'script':
        preprocessing_search       = re.compile(r'# Preprocessing([\s\S]+)#', flags=re.IGNORECASE)
        try:
            preprocessing_content  = re.search(preprocessing_search, current_file_content).group(1)
        except:
            print('Unable to identify preprocessing section of script')
            preprocessing_content  = '# Nothing to see here'

    else:
        raise ValueError('Notebooks are not supported right now')

    # Compose health functions content
    health_content = '# [1] PostgreSQL Health Check\ndef psql_health(config):\n    \'\'\'\n    PostgreSQL Health Check\n    Returns health of PostgreSQL\n\n    Input: None\n    Process: Send dummy query\n    Output: Status UP or DOWN for psql\n    \'\'\'\n    import psycopg2 as psql\n\n    try:\n        # Create connection and cursor\n        db_con = psql.connect(host     = config[\'psql_host\'],\n                              database = config[\'psql_database\'],\n                              user     = config[\'psql_user\'],\n                              password = config[\'psql_password\'],\n                              connect_timeout = 3)\n        \n        cur = db_con.cursor()\n        \n        # Execute query and fetch results\n        query_string = "SELECT call_id, call_duration, transcription_text FROM call LIMIT 1"\n        cur.execute(query_string)\n        query_call = cur.fetchall()\n        \n        # Close connection with db\n        db_con.close()\n\n        # Assign health check result\n        status = \'UP\'\n\n    except:\n        status = \'DOWN\'\n\n    finally:\n        return(status)\n\n'


    # Create main.py - where the high level structure of the application exists
    try:
        with open(os.path.join(directory,'main.py'), 'w') as main:
            main.write(main_content)
            main.close()
    except:
        raise ValueError('File main.py already exists')

    # Create preprocessing.py
    try:
        with open(os.path.join(directory,'preprocessing_functions.py'), 'w') as preprocessing:
            preprocessing.write(preprocessing_content)
            preprocessing.close()
    except:
        raise ValueError('File preprocessing.py already exists')

    # Create health_functions.py
    try:
        with open(os.path.join(directory,'health_functions.py'), 'w') as preprocessing:
            preprocessing.write(health_content)
            preprocessing.close()
    except:
        raise ValueError('File health_functions.py already exists')

    # Return results
    print('API successfully generated!')



def get_current_filename(jupyter_port=8888):
    '''
    # Get Current Filename
    
    # Description
        Get file name from file that executed this call
    This function is needed in order to read the file and split the code into modules

    # Arguments
        jupyter_port: int or string
            If called from within a jupyter notebook, the port where it is currently running, defaults to jupyter's default, 8888.

    # Examples
    
    # Credits
        Jupyter notebook section inspired by: https://github.com/jupyter/notebook/issues/1000
    '''
    import sys
    filetype = 'script'
    current_filename = sys.argv[0]

    # Check if it is a jupyter notebook
    if sys.argv[0].find('ipykernel_launcher') != -1:
        import json
        import os.path
        import re
        import ipykernel
        import requests
        from requests.compat      import urljoin
        from notebook.notebookapp import list_running_servers

        # Extract kernel id and retrieve list of active servers
        kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
        servers   = list_running_servers()

        # Send a request to each server in order to obtain more info about it
        for server in servers:
            response = requests.get("http://127.0.0.1:" + str(jupyter_port) + "/api/sessions",
                                    params={'token': server['token']})

            # If the current notebook is the desired kernel id, return path to notebook file
            for nn in json.loads(response.text):
                if nn['kernel']['id'] == kernel_id:
                    relative_path = nn['notebook']['path']
                    return(os.path.join(server['notebook_dir'], relative_path), filetype)

    else:
        return(current_filename, filetype)








def compose_api(load_ml_model_function = '', prediction_function = '', directory='application', api_model='flask-restplus', version="1.0.0", title="API Template", description = "API Template for serving machine learning models", debug=True):
    '''
    # Compose API Function - Open Deployment
    
    # Description
    Creates an API from the given code, APIs can be created from both script files as well as jupyter notebooks.

    # Arguments
        load_ml_model_function: function
            Function that loads the model from a file into memory
        prediction_function: function
            Function that outputs a prediction, it should also include
        directory: string
            Folder where the api files will be saved on.
        api_model: string
            Type of API model that will be generated, by default the API is modeled with flask-restplus
        version: string
            String of model version - useful for controlling model performance and display at the base url of the API
        title: string
            Title of the API, display at the base url of the API
        description: string
            The description of the API that will appear close to the title, display at the base url of the API
        debug: boolean
            Wether or not to activate debug mode in the API
            Defaults to True, but in production environments it is recommended to be set as False
    # Examples
    '''
    import os
    import re
    import sys
    import inspect


    # Add quotes to strings that will be used as arguments when run
    version     = "'" + str(version) + "'"
    title       = "'" + str(title) + "'"
    description = "'" + str(description) + "'"

    # Select from a library of templates
    if api_model == 'flask-restplus':
        main_content = "import gc\nimport json\nfrom   flask          import Flask, request\nfrom   flask_restplus import Resource, Api, Namespace\nimport health_functions        as health\nimport preprocessing_functions as preprocessing\n\n# Aplication initialization\napp = Flask(__name__)\napi = Api(app, version={0}, title = {1}, description = {2})\n\n\n# Import configuration parameters\nwith open('app_config.json') as config_file:\n\tconfig = json.load(config_file)\n\n\n# Singleton for model deployment\nclass Singleton(object):\n    def __new__(cls, *args, **kwargs):\n        if not hasattr(cls, '_instance'):\n            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)\n        return cls._instance\n\n\n# Assign model to the singleton\nsingle_model    = Singleton()\nsingle_model.ai = {3}\n\n\n# General Health Check\nclass Health(Resource):\n\t'''\n\tGeneral Health Check\n\tReturns operational status of application and it's components\n\n\tInput: None\n\tProcess: Checks the health of every component, usually by dummy queries\n\tOutput: Status 'UP' if accessible, or else http error code\n\t'''\n\tdef get(self):\n\t\tstatus = dict()\n\t\tstatus['Application'] = 'UP'\n\t\t#status['PostgreSQL'] = health.psql_health(config)\n\t\treturn status\n\n\n\n# Core Function: Call Analysis\nclass Predict(Resource):\n\t'''\n\tInput: Some id\n\n\tProcess: Receives a Call ID, retrieves the data associated with that call from PostgreSQL.\n\tPrepares the data and predicts using viceroy lgb.\n\n\tOutput: Returns call id, wether the call is considered a lead or not, trade in tag, financing tag, banks mentioned and execution log details.\n\t'''\n\tdef get(self, some_id):\n\t\t{4}\n\n\t\tresults = dict()\n\t\tresults['id'] \t\t\t = str(some_id)\n\t\tresults['prediction'] \t = 0.99\n\t\tresults['model_version'] = {0}\n\t\treturn(results)\n\n\n\n# Assing resources and corresponding routes\napi.add_resource(Health,  '/health')\napi.add_resource(Predict, '/predict/<string:some_id>')\n\n\n# Run application\ngc.collect()\nif __name__ == '__main__':\n\tapp.run(host = '0.0.0.0', debug = {5})\n".format(version, title, description, load_ml_model_function, prediction_function, debug)
    else:
        raise ValueError('Only flask-restplus is supported right now')


    # Create API folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print('Directory {} already exists'.format(directory))


    # Identify wether the current file is a notebook or a script
    filename, filetype   = get_current_filename()
    current_file_content = open(filename,'r').read()

    # Scan file for Preprocessing content
    if filetype == 'script':
        preprocessing_search       = re.compile(r'# Preprocessing([\s\S]+)#', flags=re.IGNORECASE)
        try:
            preprocessing_content  = re.search(preprocessing_search, current_file_content).group(1)
        except:
            print('Unable to identify preprocessing section of script')
            preprocessing_content  = '# Nothing to see here'

    else:
        raise ValueError('Notebooks are not supported right now')

    # Compose health functions content
    health_content = '# [1] PostgreSQL Health Check\ndef psql_health(config):\n    \'\'\'\n    PostgreSQL Health Check\n    Returns health of PostgreSQL\n\n    Input: None\n    Process: Send dummy query\n    Output: Status UP or DOWN for psql\n    \'\'\'\n    import psycopg2 as psql\n\n    try:\n        # Create connection and cursor\n        db_con = psql.connect(host     = config[\'psql_host\'],\n                              database = config[\'psql_database\'],\n                              user     = config[\'psql_user\'],\n                              password = config[\'psql_password\'],\n                              connect_timeout = 3)\n        \n        cur = db_con.cursor()\n        \n        # Execute query and fetch results\n        query_string = "SELECT call_id, call_duration, transcription_text FROM call LIMIT 1"\n        cur.execute(query_string)\n        query_call = cur.fetchall()\n        \n        # Close connection with db\n        db_con.close()\n\n        # Assign health check result\n        status = \'UP\'\n\n    except:\n        status = \'DOWN\'\n\n    finally:\n        return(status)\n\n'


    # Create main.py - where the high level structure of the application exists
    try:
        with open(os.path.join(directory,'main.py'), 'w') as main:
            main.write(main_content)
            main.close()
    except:
        raise ValueError('File main.py already exists')

    # Create preprocessing_functions.py
    try:
        with open(os.path.join(directory,'preprocessing_functions.py'), 'w') as preprocessing:
            preprocessing.write(preprocessing_content)
            preprocessing.close()
    except:
        raise ValueError('File preprocessing.py already exists')

    # Create health_functions.py
    try:
        with open(os.path.join(directory,'health_functions.py'), 'w') as health:
            health.write(health_content)
            health.close()
    except:
        raise ValueError('File health_functions.py already exists')

    # Return results
    print('API successfully generated!')



def compose_faas(version, load_ml_model_function, prediction_function, model_file, preprocessing_function = None, directory='.'):
    '''
    # Compose FaaS body

    # Description
    # Arguments
    # Examples
    '''
    import os
    import inspect

    # Unit tests
    assert(inspect.isfunction(load_ml_model_function), "Object 'load_ml_model_function' is not a function")
    assert(inspect.isfunction(prediction_function), "Object 'prediction_function' is not a function")

    # Create application folder
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print('Directory {} already exists'.format(directory))

    # Extract function body as string
    load_ml_code    = inspect.getsource(load_ml_model_function)
    load_ml_name    = load_ml_model_function.__name__
    prediction_code = inspect.getsource(prediction_function)
    prediction_name = prediction_function.__name__

    version = "'" + version + "'"

    if model_file:
        load_ml_name = load_ml_name + "('" + str(model_file) + "')"
    else:
        load_ml_name = load_ml_name +'()'

    prediction_name  = prediction_name + '(model = single_model.ai, dataframe = input_value)'

    # Compose contents to be written on main.py
    main_content = "import gc\nimport datetime\nimport pandas as pd\nfrom flask import jsonify\n\n# Load preprocessing functions if available\ntry:\n    from preprocessing_functions import *\nexcept:\n    pass\n\nversion = {0}\n\n# Function to load model\n{1}\n\n# Function to generate predictions\n{2}\n\n# Send to Mongo db\n# Soon\n\n# Singleton for model deployment\nclass Singleton(object):\n    def __new__(cls, *args, **kwargs):\n        if not hasattr(cls, '_instance'):\n            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)\n        return cls._instance\n\n\n# Assign model to the singleton\nsingle_model    = Singleton()\nsingle_model.ai = {3}\n\n\n# Compose result function\ndef compose_result(input_value, version):\n    result = dict()\n    result['prediction'] = {4}\n    result['metadata']   = dict([('version',str(version)),('date',str(datetime.datetime.now(datetime.timezone.utc)))])\n    return(result)\n\n\n# Clear as much memory as possible\ngc.collect()\n\n\n# Prediction Method\ndef analytical_predict(request):\n    input_value = request.get_json(force=True)\n    colnames    = input_value.keys()\n    input_value = pd.io.json.json_normalize(input_value)\n    input_value = input_value[colnames]\n    final_result = compose_result(input_value, version)\n    return(jsonify(final_result))\n\n\n\n".format(version, load_ml_code, prediction_code, load_ml_name, prediction_name)

    # Create preprocessing_functions.py - where the preprocessing functions are located
    if prediction_function:
        try:
            preprocess_code = inspect.getsource(preprocessing_function)
            with open(os.path.join(directory,'preprocessing_functions.py'), 'w') as preprocessing:
                preprocessing.write(preprocess_code)
                preprocessing.close()
        except:
            raise ValueError('File preprocessing_functions.py already exists')


    # Create main.py - where the executable function exists
    try:
        with open(os.path.join(directory,'main.py'), 'w') as main:
            main.write(main_content)
            main.close()
    except:
        raise ValueError('File main.py already exists')

    print('FaaS files successfully generated!')


def delete_files(remove_list):
    '''
    # File Deletion Function
    
    # Description
    Use with caution - will delete certain files

    # Arguments
    # Examples
    '''
    import os
    for file in remove_list:
        os.remove(file)


def deploy_faas(version, load_ml_model_function, prediction_function, model_file, preprocessing_function = None, directory='.'):
    '''
    # Generate FaaS Function

    # Description
    Creates a self contained zip file ready to be deployed as a FaaS

    # Arguments
    # Examples
    '''
    generate_freeze(extra_packages=['pandas==0.24.2'])
    compose_faas(version, load_ml_model_function, prediction_function, model_file, preprocessing_function, directory)

    if preprocessing_function:
        generate_zip(local_directory_files = ['requirements.txt', model_file, 'main.py', 'preprocessing_functions.py'])
    else:
        generate_zip(local_directory_files = ['requirements.txt', model_file, 'main.py'])

    print('yeet!')


    
def deploy(deployment_type = ['container','function'], output_type = [None, 'google_cloud_functions', 'aws_lambda']):
    # Deploy ML Model Function

    # Description
    # Arguments
    # Examples
    if deployment_type == 'container' and output_type != [None, 'google_cloud_functions', 'aws_lambda']:
        print("The argument 'output_type' is only necessary when deploying as a container ")

    if deployment_type == 'container':
        deploy_container()
    elif deployment_type == 'function':
        if output_type == 'google_cloud_functions':
            deploy_faas(provider = 'google_cloud')
        elif output_type == 'aws_lambda':
            deploy_faas(provider = 'aws_lambda')
        else:
            raise ValueError('Invalid output_type')
    else:
        raise ValueError('Invalid deployment_type')
