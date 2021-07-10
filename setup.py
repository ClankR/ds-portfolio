from setuptools import setup

setup(name='spy',
      version='0.3',
      description='Some useful python common functions.',
      url='https://github.com/ClankR/spy',
      author='Samuel Mitchell',
      license='MIT',
      install_requires = ['pyspark', 'pandas', 'plotly', 'streamlit', 
                        
                        'numpy', 'seaborn', 'matplotlib', 'requests',
                            'folium', 'geopy', 'Pillow', 'pathlib', 'plotly',
                            'boto3', 'findspark', 'pptree'],
      packages=['spy'],
      include_package_data = True,
      long_description=open('README.md').read(),
      zip_safe=False)
