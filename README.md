# iMAX FRET data analysis code
Code in this repository accompanies our manuscript on iMAX-FRET single-molecule analysis: [xxx].

*Documentation and refactoring of code are in progress and will be completed shortly.* 

## Installation
Install the required conda environment and activate it prior to use. Use the `mamba` drop-in replacement if you have it. From the git base directory:
```shell
mamba env install -f env.yaml
conda activate iMAX-FRET
```

## R0 estimation
Forster radius (R0) Can be estimated from iMAX-FRET data of dye triangles in our DNA nanostructure as described in our paper. To fit R0, run:
```shell
python R0_estimation/model_triangle_positions_hyperopt.py 
...
```

## streptavidin modeling
To fit single-molecule measurements on biotin pocket locations and visualize the results:
```shell
python streptavidin_modeling/make_strep_figure.py
...
```

## Triangle classification
For structure-aided classifier training and testing for single-molecule data, run:
```shell
python triangle_classification/run_pipeline_triangle_classifier.py 
...
```

## License
The code in this repository is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work.  If not, see <https://creativecommons.org/licenses/by-sa/4.0/>.
