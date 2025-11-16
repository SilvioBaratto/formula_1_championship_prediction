import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { NgOptimizedImage, DecimalPipe } from '@angular/common';
import { SimulationService, SimulationResponse } from '../services/simulation.service';

@Component({
  selector: 'app-championship',
  imports: [NgOptimizedImage, DecimalPipe],
  templateUrl: './championship.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChampionshipComponent {
  private simulationService = inject(SimulationService);

  protected loading = signal(false);
  protected error = signal<string | null>(null);
  protected results = signal<SimulationResponse | null>(null);

  protected topDrivers = computed(() => {
    const res = this.results();
    return res?.championship_predictions.slice(0, 10) ?? [];
  });

  protected modelTypeDisplay = computed(() => {
    const modelType = this.results()?.simulation_params.model_type;
    return modelType ? modelType.replace(/_/g, ' ').toUpperCase() : '';
  });

  protected buttonClasses = computed(() => ({
    'relative inline-flex items-center gap-3 rounded-lg px-8 py-4 text-lg font-semibold text-white shadow-xl transition-all duration-200 min-h-11': true,
    'bg-red-600 hover:bg-red-700 hover:shadow-2xl hover:scale-105': !this.loading(),
    'bg-red-600/50 cursor-not-allowed': this.loading()
  }));

  protected getDriverImagePath(driverCode: string): string {
    return `/assets/drivers/${driverCode}.webp`;
  }

  protected getPositionClasses(position: number) {
    return {
      'bg-gradient-to-r from-yellow-500/20 to-transparent border-l-4 border-yellow-500': position === 1,
      'bg-gradient-to-r from-slate-400/20 to-transparent border-l-4 border-slate-400': position === 2,
      'bg-gradient-to-r from-orange-600/20 to-transparent border-l-4 border-orange-600': position === 3,
      'hover:bg-slate-800/50': true,
      'transition-all duration-200': true
    };
  }

  protected runSimulation(): void {
    this.loading.set(true);
    this.error.set(null);

    this.simulationService.runSimulation(10000).subscribe({
      next: (response) => {
        this.results.set(response);
        this.loading.set(false);
      },
      error: (err: HttpErrorResponse) => {
        this.error.set(err.error?.detail || 'Simulation failed');
        this.loading.set(false);
      }
    });
  }
}
