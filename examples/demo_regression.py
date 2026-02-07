import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from umnn import MonotonicNN

def train_monotonic_regression(args):
    # 1. Data Generation
    # Ground truth: Monotonic function y = 0.5*x + 2*sigmoid(x)
    torch.manual_seed(42)
    N = 500
    # Inputs between -5 and 5
    x = (torch.rand(N, 1) * 10 - 5).sort(0)[0] 
    
    # Ground truth
    y_true = 0.5 * x + 2 * torch.sigmoid(x)
    
    # Add noise (making it hard to be strictly monotonic locally without constraint)
    y_noise = y_true + torch.randn(N, 1) * 0.3
    
    # Context
    # MonotonicNN requires a conditioning vector h. 
    # Since this is unconditional regression, we use a fixed zero vector.
    h = torch.zeros(N, 1)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Model Initialization
    # We use a context dimension of 1 (for the dummy context)
    model = MonotonicNN(context_dim=1, hidden_layers=[50, 50, 50], nb_steps=args.steps)
    model.to(device)
    
    x = x.to(device)
    h = h.to(device)
    y_target = y_noise.to(device)
    
    # 3. Training Loop
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        # x: (N, 1), h: (N, 1)
        y_pred = model(x, h)
        
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            
    print(f"Final Loss: {loss.item():.6f}")
    
    # 4. Visualization
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            model.eval()
            with torch.no_grad():
                # Smooth grid for plotting
                x_test = torch.linspace(-6, 6, 200).view(-1, 1).to(device)
                h_test = torch.zeros(200, 1).to(device)
                y_pred_smooth = model(x_test, h_test).cpu().numpy()
                
            x_cpu = x.cpu().numpy()
            y_cpu = y_noise.cpu().numpy()
            y_true_cpu = y_true.cpu().numpy()
            x_test_cpu = x_test.cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(x_cpu, y_cpu, alpha=0.4, color='gray', label='Noisy Training Data')
            plt.plot(x_cpu, y_true_cpu, '--', color='blue', label='Ground Truth')
            plt.plot(x_test_cpu, y_pred_smooth, color='red', linewidth=2, label='UMNN Prediction')
            
            plt.title('UMNN Monotonic Regression')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            out_file = "umnn_regression_demo.png"
            plt.savefig(out_file)
            print(f"\nPlot saved to {out_file}")
            
        except ImportError:
            print("\n[WARN] matplotlib not installed. Skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMNN Regression Demo")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=20, help="Number of integration steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting")
    parser.set_defaults(plot=True)
    
    args = parser.parse_args()
    train_monotonic_regression(args)
