
def test_side_classificator(model, dataloader):
    total_samples = 0
    wrong_predictions = 0

    for image_paths, classes, _ in dataloader:
        features = model.reliefF(image_paths, 'side')
        
        if features is None:
            raise KeyError('Lack of features')

        side_labels = [1 if cls == 'revers' else 0 for cls in classes[0]['side']]

        features_scaled = model.side_scaler.transform(features)

        side_preds = model.side_classifier.predict(features_scaled)

        batch_errors = sum(p != l for p, l in zip(side_preds, side_labels))
        wrong_predictions += batch_errors
        total_samples += len(side_labels)

        print(f"Labels: {side_labels}")
        print(f"Predictions: {side_preds}")
        print(f"Batch errors: {batch_errors}")

    error_rate = wrong_predictions / total_samples * 100
    print(f"Total wrong predictions: {wrong_predictions}/{total_samples} ({100 - error_rate:.2f}%)")