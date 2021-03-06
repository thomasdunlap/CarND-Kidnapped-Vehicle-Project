/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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

  // Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 50;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> d_x(0, std[0]);
  normal_distribution<double> d_y(y, std[1]);
  normal_distribution<double> d_theta(0, std[2]);

  for (int i = 0; i < num_particles; i++) {

    // Define Particle object's attributes
    Particle p;
    p.id = i;
    p.x = x + d_x(gen);
    p.y = y + d_y(gen);
    p.theta = theta + d_theta(gen);
    p.weight = 1.0;

    // Append particle to particles vector
    particles.push_back(p);

    // Append p weight to weight vector
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> d_x(0, std_pos[0]);
  normal_distribution<double> d_y(0, std_pos[1]);
  normal_distribution<double> d_theta(0, std_pos[2]);

  // Like Python's `for x in container:`
  for (auto& particle : particles) {
    // Normalize for yaw_rate approaching zero
    if (yaw_rate < .001) {
      particle.x += velocity * cos(particle.theta)*delta_t + d_x(gen);
      particle.y += velocity * sin(particle.theta)*delta_t + d_y(gen);
    } else {
      particle.x += velocity/yaw_rate*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + d_x(gen);
      particle.y += velocity/yaw_rate*(-cos(particle.theta + yaw_rate*delta_t) + cos(particle.theta)) + d_y(gen);
    }
    // Normalize theta out of range
    particle.theta += yaw_rate*delta_t + d_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs>& predicted,
                                     vector<LandmarkObs>& observations,
                                     Particle& particle) {

  // Find the predicted measurement that is closest to each observed measurement and assign the
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	// implement this method and use it as a helper during the updateWeights phase.

  // Pair vectors for easy obs, actual storage
  vector<pair<double, double>> sense_x;
  vector<pair<double, double>> sense_y;
  vector<int> associations;

  for (auto& observation : observations) {
    double min_dist = numeric_limits<double>::max(); // initialise with "infinity"
    // Closest landmark
    LandmarkObs closest_lmrk{0, 0.0, 0.0};
    for (auto& landmark : predicted) {
      double distance = dist(observation.x, observation.y, landmark.x, landmark.y);
      if (distance < min_dist) {
        min_dist = distance;
        closest_lmrk = landmark;
      }
    }
    // emplace_back because push_back T doesn't work with pair vectors
    sense_x.emplace_back(observation.x, closest_lmrk.x);
    sense_y.emplace_back(observation.y, closest_lmrk.y);
    associations.push_back(closest_lmrk.id);

  }
  // Setter for sense_x
  SetAssociations(particle, associations, sense_x, sense_y);

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  for (auto& particle : particles) {

    // Step 1:
    // create a vector of observations converted to a global reference frame
    // and transform them to map coordinates:
    vector<LandmarkObs> map_observations;
    for (auto& observation : observations) {

      LandmarkObs map_observation{0, 0.0, 0.0};
      map_observation.x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
      map_observation.y = particle.y + observation.y * cos(particle.theta) + observation.x * sin(particle.theta);
      map_observations.push_back(map_observation);
    }


    // for a given particle create a vector of landmarks within a range of our sensor
    // and if the landmark is within the sensor range we add it to a landmarks vector
    vector<LandmarkObs> landmarks;
    for (auto& map_landmark : map_landmarks.landmark_list) {
      LandmarkObs landmark{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f};
      if (dist(particle.x, particle.y, landmark.x, landmark.y) < sensor_range) {
        landmarks.push_back(landmark);
      }
    }

    // Associations between landmarks and measurements
    dataAssociation(landmarks, map_observations, particle);

    particle.weight = 1.0;

    double C =  1.0 / (2.0*M_PI * std_landmark[0]*std_landmark[1]);

    for (int i=0; i < particle.sense_x.size(); ++i) {

      double x_2 = (particle.sense_x[i].first - particle.sense_x[i].second) * (particle.sense_x[i].first - particle.sense_x[i].second);
      double y_2 = (particle.sense_y[i].first - particle.sense_y[i].second) * (particle.sense_y[i].first - particle.sense_y[i].second);

      double ex = -x_2/(2 * std_landmark[0]*std_landmark[0]) - y_2/(2 * std_landmark[1]*std_landmark[1]);

      particle.weight *= C * exp(ex);
    }

  }
  // Update weights
  for (int i=0; i < num_particles; i++) {
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {

  // Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<int> d(weights.begin(), weights.end());

  for (auto& particle : particles) {
    particle = particles[d(gen)];
  }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations,
                                     const vector<pair<double, double>>& sense_x, const vector<pair<double, double>>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseX(Particle best)
{

  vector<double> v;
  for (auto& vi : best.sense_x) {
    v.push_back(vi.first);
  }

	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v;
  for (auto& vi : best.sense_y) {
    v.push_back(vi.first);
  }

  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
