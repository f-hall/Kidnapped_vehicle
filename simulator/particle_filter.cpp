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
    // the random generator for the three normal distributions (x, y, theta)
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    // the number of the particles and setting particles/weights up
    num_particles = 500;
    particles = std::vector<Particle>(num_particles);
    weights = std::vector<double>(num_particles);

    // Give each particle necessary Information
    for (int i = 0; i < num_particles; ++i)
        {
          particles[i].x = dist_x(gen);
          particles[i].y = dist_y(gen);
          particles[i].theta = dist_theta(gen);
          particles[i].weight = 1.0;
          weights[i] = particles[i].weight;
          particles[i].id = i;
        }
        // Setting init-value to true
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    std::default_random_engine gen;

    // Predict for each particle, where it's next position (after timestep) should be
    for (int i = 0; i < num_particles; ++i)
    {
        // Use two different models (one if the yaw rate is around zero)
        if (yaw_rate > 0.0001 || yaw_rate < -0.0001)
        {
            particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta+delta_t*yaw_rate)-sin(particles[i].theta));
            particles[i].y = particles[i].y + (velocity/yaw_rate) * (-cos(particles[i].theta+delta_t*yaw_rate)+cos(particles[i].theta));
        }
        else
        {
            particles[i].x = particles[i].x + velocity*cos(particles[i].theta)*delta_t;
            particles[i].y = particles[i].y + velocity*sin(particles[i].theta)*delta_t;
        }
        particles[i].theta = particles[i].theta + delta_t*yaw_rate;

        // use normal distribution again for the uncertainty of measurements
        std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);

    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Use nearest neighbor to find the best observation for each predicted landmark
    if (predicted.size() == 0)
    {
        return;
    }
    else
    {
        for (auto& obs : observations)
        {
            double minimum = dist(obs.x, obs.y, predicted[0].x, predicted[0].y);
            obs.id = predicted[0].id;
            for (auto pred : predicted)
            {
                double dist_help = dist(obs.x, obs.y, pred.x, pred.y);
                if (dist_help < minimum)
                {
                    obs.id = pred.id;
                    minimum = dist_help;
                }
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
        std::vector<LandmarkObs> landmarks_all;

        // Reset weights
        double weight_all = 0;

        for (int i = 0; i < num_particles; ++i){
        particles[i].weight = 1;
        }

        // Get list of all landmarks
        for (auto map_land : map_landmarks.landmark_list)
        {
            landmarks_all.push_back(LandmarkObs{map_land.id_i-1, map_land.x_f, map_land.y_f});
        }

        int counter = 0;

        // Update weight for each particle in this loop
        for (auto& p : particles)
        {
        std::vector<LandmarkObs> observation = observations;
            std::vector<LandmarkObs> landmarks;
            landmarks.clear();
            int l_count = 0;
            // Chose only landmarks, which are in sensor_range of the particle - this speeds nearest neighbor up
            for (LandmarkObs local_land : landmarks_all)
            {
                if (p.x -(sensor_range*1.1) < local_land.x && local_land.x < p.x + (sensor_range*1.1) && p.y -(sensor_range*1.1) < local_land.x && local_land.y < p.y + (sensor_range*1.1))
                {
                    landmarks.push_back(LandmarkObs{l_count, local_land.x, local_land.y});
                    l_count = l_count+1;
                }

            }

            // Change observations from car-coordinates to global-coordinates
            for (LandmarkObs& obs : observation)
            {
                double obx = obs.x;
                double oby = obs.y;
                obs.x = cos(p.theta)*obx - sin(p.theta)*oby + p.x;
                obs.y = sin(p.theta)*obx + cos(p.theta)*oby + p.y;
            }

            // use dataAssociation(with NN) from above
            dataAssociation(landmarks, observation);

            // Update weights of the particle for each observation (using Gauss)
            for (auto obs : observation)
            {
                double x_land = landmarks[obs.id].x;
                double y_land = landmarks[obs.id].y;
                double x_obs = obs.x;
                double y_obs = obs.y;

                double x_diff = pow(x_obs-x_land,2)/(2*pow(std_landmark[0],2));
                double y_diff = pow(y_obs-y_land,2)/(2*pow(std_landmark[0],2));

                p.weight *= 1/(2*M_PI*std_landmark[0]*std_landmark[1])* exp(-(x_diff + y_diff));

            }

            weight_all = weight_all + p.weight;

        }

        // Setting weights to particles
        for (int i = 0; i < num_particles; ++i)
        {
            particles[i].weight = particles[i].weight/weight_all;
            weights[i] = particles[i].weight;
        }
}

void ParticleFilter::resample() {

    // Resample particles, using a discrete distribution and a random_engine
    std::default_random_engine gen;
    std::vector<Particle> resampled_particles;

    std::discrete_distribution<> d(weights.begin(), weights.end());

    for(int i=0; i < num_particles; ++i)
    {
        resampled_particles.push_back(particles[d(gen)]);
        resampled_particles[i].id = i;
    }
    particles = resampled_particles;

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
