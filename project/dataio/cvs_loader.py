from project.dataio.csv_loader import CSVDataset
ds = CSVDataset("toy", root="data")
dl = DataLoader(ds, batch_size=128, shuffle=True)
