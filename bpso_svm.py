
# BPSO-SVM ENGINE: MASTER EXECUTION LOOP
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from preprocess import get_preprocessed_pipeline

class BulletproofBPSO:
    def __init__(self, num_particles=10, max_iter=5, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def _sigmoid(self, velocity):
        return 1 / (1 + np.exp(-velocity))
        
    def optimize(self, X_train, y_train, feature_names):
        num_features = X_train.shape[1]
        print(f"[*] Launching Swarm: {self.num_particles} particles over {num_features} atmospheric dimensions.")
        
        position = np.random.randint(2, size=(self.num_particles, num_features))
        velocity = np.zeros((self.num_particles, num_features))
        
        for i in range(self.num_particles):
            if np.sum(position[i]) == 0: position[i, np.random.randint(num_features)] = 1
                
        pbest_pos = np.copy(position)
        pbest_fitness = np.zeros(self.num_particles) - 1.0
        gbest_pos = np.zeros(num_features)
        gbest_fitness = -1.0
        
        for iteration in range(self.max_iter):
            print(f"\n[*] Epoch {iteration + 1}/{self.max_iter} executing...")
            
            for i in range(self.num_particles):
                active_features = np.where(position[i] == 1)[0]
                
                if len(active_features) == 0:
                    current_fitness = -1.0
                else:
                    X_subset = X_train[:, active_features]
                    
                    # THE FIX: Stratified slice to prevent scalar divide zero-errors
                    try:
                        sample_size = min(5000, X_subset.shape[0])
                        _, X_sample, _, y_sample = train_test_split(
                            X_subset, y_train, 
                            test_size=sample_size, 
                            stratify=y_train, 
                            random_state=42
                        )
                        
                        # Train max-margin SVM with class balancing
                        clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
                        clf.fit(X_sample, y_sample)
                        
                        y_pred = clf.predict(X_sample)
                        current_fitness = matthews_corrcoef(y_sample, y_pred)
                        
                        if np.isnan(current_fitness): current_fitness = -1.0
                    except Exception:
                        current_fitness = -1.0
                
                if current_fitness > pbest_fitness[i]:
                    pbest_fitness[i] = current_fitness
                    pbest_pos[i] = np.copy(position[i])
                    
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = np.copy(position[i])
            
            print(f"    -> Current Global Best MCC: {gbest_fitness:.4f}")
            
            # Trajectory updates
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
                velocity[i] = (self.w * velocity[i] + 
                               self.c1 * r1 * (pbest_pos[i] - position[i]) + 
                               self.c2 * r2 * (gbest_pos - position[i]))
                sq_prob = self._sigmoid(velocity[i])
                position[i] = np.where(np.random.rand(num_features) < sq_prob, 1, 0)
                if np.sum(position[i]) == 0: position[i, np.random.randint(num_features)] = 1
                    
        selected_indices = np.where(gbest_pos == 1)[0]
        selected_features = [feature_names[idx] for idx in selected_indices]
        
        print("\n[+] CONVERGENCE ACHIEVED!")
        print(f"    -> Optimized Feature Set Size: {len(selected_features)} features.")
        print(f"    -> Selected Variables: {selected_features}")
        print(f"    -> Final MCC Fitness: {gbest_fitness:.4f}")
        return selected_indices, selected_features

