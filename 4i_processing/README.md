```bash
# install phenoscapes
# setup conda environment
conda create -n phenoscapes python=3.10
git clone https://github.com/quadbio/morphodynamics_human_brain_organoid
cd morphodynamics_human_brain_organoid/4i_processing
pip install .  
```

```bash
# run retina example
cd morphodynamics_human_brain_organoid/4i_processing/example_data
python run_retina_example.py
```