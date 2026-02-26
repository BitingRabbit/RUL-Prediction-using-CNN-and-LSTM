"""Dieses Modul enthält Funktionen zum Laden eines gespeicherten Modells, 
Berechnung des MSE-Verlusts und Evaluierung des Modells auf dem Testdatensatz."""

# pylint: disable=import-error
# pylint Fehler, da die Module richtig geladen werden
import torch
from sklearn.metrics import accuracy_score, r2_score
from torch import Tensor
from torch import nn


def load_model(model, checkpoint_path, device):
    """
    Lädt ein gespeichertes Modell aus einem Checkpoint.

    Args:
        model (nn.Module): Das Modell, in das die Gewichte geladen werden
        checkpoint_path (str): Pfad zur gespeicherten Checkpoint-Datei (.pth)
        device (torch.device): Gerät, auf dem das Modell ausgeführt wird (CPU/MPS)

    Returns:
        nn.Module: Das Modell mit geladenen Gewichten im Evaluierungsmodus.
    """
    # letztes Modell laden
    checkpoint: dict = torch.load(checkpoint_path, map_location=device)
    # Modellgewichte laden
    model.load_state_dict(checkpoint["model_state_dict"])
    # Modell auf das angegebene Gerät verschieben
    model: nn.Module = model.to(device)
    # Modell in den Evaluierungsmodus setzen
    model.eval()
    return model


def mse_torch(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Berechnet den Mean Squared Error (MSE) zwischen wahren und vorhergesagten Werten

    Args:
        y_true (Tensor): Die tatsächlichen RUL-Werte
        y_pred (Tensor): Die vom Modell vorhergesagten RUL-Werte

    Returns:
        Tensor: Der mittlere quadratische Fehler als skalarer Tensor
    """
    return torch.mean((y_true - y_pred) ** 2)


def evaluate_test_loss(model, device, test_loader) -> float:
    """
    Evaluiert das Modell auf dem Testdatensatz und berechnet MSE, R²-Score und Accuracy

    Args:
        model (nn.Module): Das trainierte Modell im Evaluierungsmodus
        device (torch.device): Gerät, auf dem das Modell ausgeführt wird (CPU/MPS)
        test_loader (DataLoader): DataLoader mit den Testdaten

    Returns:
        tuple[float, float, float]: Durchschnittlicher MSE, R²-Score und Accuracy
    """
    all_y, all_pred = [], []
    # Modell im Evaluierungsmodus (sollte bereits durch load_model gesetzt sein)
    model.eval()
    total_loss: float = 0.0
    num_batches: int = 0
    # Forward-Pass durch den Testdatensatz ohne Gradientenberechnung

    # pylint: disable=invalid-name
    # Variablenname X_batch entsprechend gewählt und groß geschrieben, um zu verdeutlichen,
    # dass es sich um einen Batch von Eingabedaten handelt
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch: Tensor = X_batch.to(device).unsqueeze(1)  # (B, 1, 10, 36)
            y_batch: Tensor = y_batch.to(device).view(-1, 1)

            # Prediction des Modells
            output: Tensor = model(X_batch)

            # Berechnung des MSE-Verlusts für die aktuelle Batch
            loss: Tensor = mse_torch(y_batch, output)
            # Summieren des Gesamtverlusts über alle Batches
            total_loss += loss.item()

            all_y.append(y_batch.detach().cpu())
            all_pred.append(output.detach().cpu())

            num_batches += 1

    # Alle gesammelten wahren und vorhergesagten Werte zusammenführen
    all_y = torch.cat(all_y)
    # Alle gesammelten vorhergesagten Werte zusammenführen
    all_pred = torch.cat(all_pred)
    # Durchschnittlichen Verlust über alle Batches berechnen
    avg_loss: float = total_loss / max(num_batches, 1)
    # R²-Score und Accuracy berechnen
    r2: float = r2_score(all_y.numpy(), all_pred.numpy())
    accuracy: float = accuracy_score(all_y.numpy(), all_pred.round().numpy())

    # Speicher freigeben
    del all_y, all_pred

    return (avg_loss, r2, accuracy)
