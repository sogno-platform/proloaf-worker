from setuptools import setup, find_packages

setup(
    name="sogno-forecasting-worker",
    version="0.1",
    description="Classes for interfacing with AMQP",
    author="Florian Oppermann",
    url="https://github.com/sogno-platform/proloaf-worker",
    packages=find_packages()  ,#["sognoforecasting"],
    license="Apache-2.0 License",
    python_requires=">=3.8",
    install_requires=[
        "pydantic ~= 1.9",
        "redis ~= 4.1",
        "sogno-job-queue @ git+https://github.com/sogno-platform/sogno-job-queue@develop#egg=sogno-job-queue",
        "proloaf @ git+https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/@prep-for-sogno-service-new-history#egg=proloaf",
    ],
)
