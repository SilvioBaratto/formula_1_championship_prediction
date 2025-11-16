import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface DriverStanding {
  driver: string;
  expected_position: number;
  prob_win: number;
  prob_podium: number;
  expected_points: number;
  current_points: number;
}

export interface WinnerProbability {
  driver: string;
  probability: number;
}

export interface SimulationParams {
  n_simulations: number;
  remaining_races: number;
  completed_races: number;
  model_type: string;
}

export interface SimulationResponse {
  championship_predictions: DriverStanding[];
  winner_probabilities: WinnerProbability[];
  simulation_params: SimulationParams;
}

@Injectable({
  providedIn: 'root',
})
export class SimulationService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api/v1';

  runSimulation(nSimulations: number = 10000): Observable<SimulationResponse> {
    return this.http.post<SimulationResponse>(
      `${this.apiUrl}/simulate?n_simulations=${nSimulations}`,
      {}
    );
  }
}
