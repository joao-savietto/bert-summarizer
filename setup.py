from setuptools import setup, find_packages

setup(
    name='bertsummarizer',
    version='1.0.0',    
    description='Extractive summarization using BERT and k-means',
    url='',
    author='joaosavietto',
    author_email='jvsavietto6@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    packages=find_packages(),
    install_requires = ['nltk==3.7', 'kneed==0.8.1', 'scikit-learn==1.0.2', 'sentence-transformers==2.2.0', 'wordcloud==1.8.2.2'],
    setup_requires = ['nltk==3.7', 'kneed==0.8.1', 'scikit-learn==1.0.2', 'sentence-transformers==2.2.0', 'wordcloud==1.8.2.2'],


    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: Microsoft :: Windows'
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'

    ],

)