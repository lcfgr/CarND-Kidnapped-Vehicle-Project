/*
 * particle_filter.cpp
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


  num_particles = 150;

  default_random_engine gen;

  // Create normal (Gaussian) distributions for x,y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Create randomized particles
  for (int i = 0; i < num_particles; ++i) {
    //randomize particle & initialise particle weight
    Particle sample_p;
    sample_p.id = i;
    sample_p.x = dist_x(gen);
    sample_p.y = dist_y(gen);
    sample_p.theta = dist_theta(gen);
    sample_p.weight = 1.0;
    //store particle
    particles.push_back(sample_p);
  }
  is_initialized = true;
  cout << "Initialized " << num_particles << " Particles, with mean values (x,y): (" <<x<< "," <<y<<"),"
          << "and mean theta: " << theta<< endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  // Create normal (Gaussian) distributions used as noise, with mean value zero (0)
  normal_distribution<double> dist_x(0,std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Predict the state of the vehicle
  for (int i=0; i< num_particles; i++) {

    // Seperate yaw values near zero (0)
    if (fabs(yaw_rate > 0.00001)){

      double theta = particles[i].theta;
      double d_theta = theta + (yaw_rate*delta_t);

      particles[i].x += (velocity/yaw_rate)*( sin(d_theta)-sin(theta) )   + dist_x(gen);
      particles[i].y += (velocity/yaw_rate)*( cos(d_theta)-cos(d_theta) ) + dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    }
      else {
      double theta = particles[i].theta;
      particles[i].x += velocity * delta_t * cos(theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(theta) + dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    }

  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i =0; i < observations.size();i++){
    double min_dist = 1000000;
    LandmarkObs temp_observation = observations[i];

    // Find the closest prediction
    //for (int j=0; j< predicted.size();j++){
    for(LandmarkObs temp_prediction:predicted){
      double temp_distance = dist(temp_observation.x, temp_observation.y, temp_prediction.x, temp_prediction.y);
      if (temp_distance < min_dist){
        min_dist = temp_distance;
        observations[i].id = temp_prediction.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // for all particles
  for (int i = 0; i < particles.size(); i++) {


    // Convert all observations to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (auto tmp_observation : observations) {
      LandmarkObs tmp_transformed;
      tmp_transformed.id = tmp_observation.id;
      tmp_transformed.x = tmp_observation.x * cos(particles[i].theta)
                          - tmp_observation.y * sin(particles[i].theta) + particles[i].x;
      tmp_transformed.y = tmp_observation.x * sin(particles[i].theta)
                          + tmp_observation.y * cos(particles[i].theta) + particles[i].y;
      transformed_observations.push_back(tmp_transformed);
    }

    // Keep the map landmarks which are within sensor range
    vector<LandmarkObs> predicted_landmarks;
    for (auto tmp_map : map_landmarks.landmark_list) {
      LandmarkObs tmp_predicted;
      tmp_predicted.x = tmp_map.x_f;
      tmp_predicted.y = tmp_map.y_f;
      tmp_predicted.id = tmp_map.id_i;
      if (dist(tmp_predicted.x, tmp_predicted.y, particles[i].x, particles[i].y) <= sensor_range) {
        predicted_landmarks.push_back(tmp_predicted);
      }
    }

    // Associate the measurements with the corresponding nearest landmarks (transformed_observations.id->nearest id)
    dataAssociation(predicted_landmarks, transformed_observations);

    particles[i].weight = 1.0;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double gauss_norm = (1 / (2 * M_PI * std_x * std_y));

    for (auto tmp_observation:transformed_observations) {

      for (int j = 0; j < predicted_landmarks.size(); j++) {
        if (predicted_landmarks[j].id == tmp_observation.id) {
          double error_x = (predicted_landmarks[j].x - tmp_observation.x);
          double error_y = (predicted_landmarks[j].y - tmp_observation.y);
          gauss_norm *= exp(-1 * (error_x * error_x / (2 * std_x * std_x)));
          gauss_norm *= exp(-1 * (error_y * error_y / (2 * std_y * std_y)));
          break; //no need to continue searching
        }
      }
    }
    particles[i].weight *= gauss_norm;
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Read all current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

    discrete_distribution<int> weights_dist(weights.begin(), weights.end());
    vector<Particle> new_particles;

    default_random_engine gen;
    // resample particles
    for ( int i = 0; i < num_particles; ++i)
      new_particles.push_back(particles[weights_dist(gen)]);

    //new set of particles(resampled)
    particles = new_particles;
  }

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
