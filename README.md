# GNNR

Code for the technical report [].

## Quick Start

Here, we provide a quick start guide on how to reproduce the results.

### Download the project and install the required dependencies.

python==3.8.16
torch==1.11.0
torchmetrics==1.3.0
dgl==0.9.1
ogb==1.3.6
shortuuid==1.0.11
pandas==2.0.3
gensim==4.3.2
numpy==1.24.3
tqdm==4.65.0
tensorboardx==2.6.2.2
scipy==1.10.1
scikit-learn==1.3.0
networkx==3.1

### Run the Code
cd Node-classification
cd CLGNN-main
To reproduce the results on the Citeseer dataset, input:
    `` python main_rphgnn.py  --dataset "citeseer" --method "rphgnn" --use_nrl False --use_label False --even_odd "all"  --train_strategy "cl" --use_input True --input_drop_rate 0.0 --drop_rate 0.5 --hidden_size 256 --squash_k 3 --num_epochs 5000 --max_patience 50 --embedding_size 512 --use_all_feat True --output_dir "outputs/citeseer/" --gpus 0 --seed 7 ``

## License

 is released under the MIT license. Further details can be found [here](LICENSE).
