import json

# Load JSON data from a file
with open("res.json", "r") as file:
    data = json.load(file)


# Define the function to create markdown tables from the provided data
def create_markdown_table(data, error_type,timer:int=1):
    # Define the datasets and the headers for the tables
    datasets = {
        "Replica": [
            "room0",
            "room1",
            "room2",
            "office0",
            "office1",
            "office2",
            "office3",
            "office4",
        ],
        "TUM": [
            "freiburg1_desk",
            "freiburg1_desk2",
            "freiburg1_room",
            "freiburg2_xyz",
            "freiburg3_long_office_household",
        ],
    }
    headers = {
        "Replica": "| Methods | Avg. | R0 | R1 | R2 | Of0 | Of1 | Of2 | Of3 | Of4 |",
        "TUM": "| Methods | Avg. | fr1/desk | fr1/desk2 | fr1/room | fr2/xyz | fr3/off. |",
    }
    separator = {
        "Replica": "|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|",
        "TUM": "|---------|-------|----------|-----------|---------|--------|---------|",
    }

    for dataset, locations in datasets.items():
        print(f"::: {{.table}}")
        print(
            f":{dataset}[@sturmBenchmarkEvaluationRGBD2012] ({error_type} RMSE ↓\[cm\]).\n"
        )
        print(headers[dataset])
        print(separator[dataset])  # Print the separator line

        # Prepare to collect data for 'ours' and calculate averages for others
        methods_data = {"ours": []}
        methods_order = []

        # Gather data from each location for all methods
        for location in locations:
            for method, values in data[dataset][location].items():
                if method not in methods_data:
                    methods_data[method] = []
                    methods_order.append(method)
                methods_data[method].append(values[error_type])

        # Calculate averages and print each method's data
        for method in methods_order:
            avg = sum(methods_data[method]) / len(methods_data[method])
            method_line = (
                f"| {method} | {avg*timer:.5f} | "
                + " | ".join(f"{x*timer:.5f}" for x in methods_data[method])
                + " |"
            )
            print(method_line)

        # Print 'ours' method data
        ours_avg = sum(methods_data["ours"]) / len(methods_data["ours"])
        ours_line = (
            f"| Ours | {ours_avg*timer:.5f} | "
            + " | ".join(f"{x*timer:.5f}" for x in methods_data["ours"])
            + " |"
        )
        print(ours_line)
        print(":::\n")


# Generate markdown tables for ATE and AAE
create_markdown_table(data, "ATE",100)
create_markdown_table(data, "AAE")
