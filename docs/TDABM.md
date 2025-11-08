## TDABM (Topological Data Analysis Ball Mapper)

CRISP-T implements the Topological Data Analysis Ball Mapper (TDABM) algorithm based on **Rudkin and Dlotko (2024)**. TDABM provides a model-free method to visualize multidimensional data and uncover hidden, global patterns in complex, noisy, or high-dimensional datasets.

### How TDABM Works

1. **Point Cloud Creation**: Data is transformed into a point cloud where each axis represents one of the selected variables (X variables).
2. **Ball Covering**: The algorithm randomly selects landmark points and creates balls of a specified radius around them, covering all data points.
3. **Connection Mapping**: Landmark points with non-empty intersections are connected, revealing the topological structure of the data.
4. **Visualization**: The result is visualized as a 2D graph where:
   - Circle size represents the number of points in each ball
   - Circle color represents the mean value of the outcome variable (Y), ranging from red (low) to purple (high)
   - Lines connect overlapping balls, showing the data's topological structure

### Using TDABM

```bash
# Perform TDABM analysis
crispt --inp corpus_dir --tdabm satisfaction:age,income,education:0.3 --out corpus_dir

# Visualize TDABM results
crispviz --inp corpus_dir --tdabm --out visualizations
```

### When to Use TDABM

- Discovering hidden patterns in multidimensional data
- Visualizing relationships between multiple variables
- Identifying clusters and connections in complex datasets
- Performing model-free exploratory data analysis
- Understanding global structure in high-dimensional data

### Reference

Rudkin, S., & Dlotko, P. (2024). Topological Data Analysis Ball Mapper for multidimensional data visualization. *Paper reference to be added - algorithm implementation based on the TDABM methodology described by the authors.*