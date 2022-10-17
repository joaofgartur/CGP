## Cartesian Genetic Programming

**Configs**

The configs required are in the *configs.json* file. You can change them there.

  * **image_width**: Width of the image in pixels.
  * **image_height**: Height of the image in pixels.
  * **num_rows**: Number of rows of the graph.
  * **num_columns**: Number of columns of the graph.
  * **level_back**: Number of how far back a node can connect to.
  * **num_input**: Number of input nodes of the graph.
  * **num_output**: Number of output nodes of the graph.
  * **num_functions**: Number of available functions.
  * **mutation_rate**: Mutation rate.
  * **max_generation**: Limit of generations to evolve.
  * **seed**: Value for random seed.

**Requirements**

| Name       | Version |
|------------|---------|
 | python     | 3.10.4  |
| imageio    | 2.19.3  |
| numpy      | 1.23.1  |
 | matplotlib | 3.5.2   |

**Execute**

To execute the script from command line use:

`python main.py <path_to_configs_file>`

**Evaluation**

Individuals can be evaluated from 0 to 100.
To evaluate a generation, follow the format:

`<individualA>=<score>,<individualB>=<score>,...`

You can evaluate all the individuals, only some of them or none. Default fitness is 0.0.