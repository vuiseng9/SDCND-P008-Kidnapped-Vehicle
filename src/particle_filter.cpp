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
#include <iomanip>

#include "particle_filter.h"

using namespace std;

ParticleFilter::ParticleFilter(unsigned int N) {
    num_particles = N;
    is_initialized = false;
    cout << "[Info]: " << num_particles << " particles created." << endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    double std_x     = std[0];
    double std_y     = std[1];
    double std_theta = std[2];

    // distributions for x, y, and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    // initialize attributes of each particle
    for (int i=0; i<num_particles; i++) {
        Particle p;
        p.id    = i;
        p.x     = dist_x(gen);
        p.y     = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight= 1.0f;

        // append particle and its weight to respective list
        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double v, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian uncertainty.
	// NOTE: When adding uncertainty you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double std_x        = std_pos[0];
    double std_y        = std_pos[1];
    double std_theta    = std_pos[2];

  	// predict motion
  	for (int i=0; i<num_particles; i++) {

    	double theta = particles[i].theta;

        if (fabs(yaw_rate) < 0.00001) { 
            // zero yaw rate
            particles[i].x += v*dt * cos(theta);
            particles[i].y += v*dt * sin(theta);
        } else {
            // non-zero yaw rate
            particles[i].x += v/yaw_rate * ( sin(theta + yaw_rate*dt) - sin(theta) );
            particles[i].y += v/yaw_rate * ( cos(theta) - cos(theta + yaw_rate*dt) );
            particles[i].theta += yaw_rate * dt;
        }

        // create distribution with input std.dev and predicted motion values as mean 
  	    normal_distribution<double> dist_x(particles[i].x, std_x);
  	    normal_distribution<double> dist_y(particles[i].y, std_y);
  	    normal_distribution<double> dist_theta(particles[i].theta, std_theta);
        
        // generate value
        particles[i].x      = dist_x(gen);
        particles[i].y      = dist_y(gen);
        particles[i].theta  = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double nearest_dist, dist;

    for (int k=0; k<observations.size(); k++)
    {
        nearest_dist = 999999999;

        for (int j=0; j<predicted.size(); j++)
        {
            dist = calc_dist(predicted[j].x, predicted[j].y, observations[k].x, observations[k].y);

            if (dist < nearest_dist) {
                nearest_dist = dist;
                observations[k].id = predicted[j].id;
            }
        }

        cout << "   xobs " << k << ", id is assigned with lm id " << observations[k].id << endl;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // declare temp variables
    double px, py, p_theta, nearest_dist, dist, sig_x, sig_y, norm, exponent, weight_norm;
    LandmarkObs lm, obs, xobs;

    // lists to store (1) landmarks in the sensor range and (2) transformed observations in map reference
    vector<LandmarkObs> lm_inrange, x_obs;

    sig_x = std_landmark[0];
    sig_y = std_landmark[1];

    weight_norm = 0; //normalizer a.k.a sum of all particles weight

    for (int i=0; i<num_particles; i++) {
        px = particles[i].x;
        py = particles[i].y;
        p_theta = particles[i].theta;

        cout << "\n[Info] P" << i << ", particle.id: " << particles[i].id <<
            ", x: " << px <<
            ", y: " << py <<
            ", theta: " << p_theta << endl;

        // reset list
        lm_inrange.clear();
        x_obs.clear();

        // filter landmarks that are within range of particle sensor
        for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
            lm.id = map_landmarks.landmark_list[j].id_i;
            lm.x  = map_landmarks.landmark_list[j].x_f;
            lm.y  = map_landmarks.landmark_list[j].y_f;

            if (calc_dist(px, py, lm.x, lm.y) < sensor_range) { 
				double _dist = calc_dist(px, py, lm.x, lm.y);	
                lm_inrange.push_back(lm);
			}
        }

        cout << "[Info] # Landmarks within sensor range: " << lm_inrange.size() 
             << " out of initial " << map_landmarks.landmark_list.size() << endl;

        // transform vehicle obs to map coord
		for (int k=0; k<observations.size(); k++) 
		{
            // set transformed obs id to invalid
            xobs.id = -1;
            
            // obs w.r.t to vehicle coord
            obs.x  = observations[k].x;
            obs.y  = observations[k].y;
	
			// translate, rotate
            xobs.x = obs.x * cos(p_theta) - obs.y * sin(p_theta) + px;
            xobs.y = obs.x * sin(p_theta) + obs.y * cos(p_theta) + py;
            
            // append transformed obs to the list
            x_obs.push_back(xobs);
 		}

        // associate x_obs to the nearest lm_inrange
        dataAssociation(lm_inrange, x_obs);

        // reset particle weight
        particles[i].weight = 1.0;

		// incrementally update weight when matching xobs id in filtered landmark list
        for (int k=0; k<x_obs.size(); k++)
        { 
			xobs.id = x_obs[k].id;
            xobs.x  = x_obs[k].x;
            xobs.y  = x_obs[k].y;

            // find matching id
        	for (int j=0; j<lm_inrange.size(); j++) {
            	if (lm_inrange[j].id == xobs.id) {
                    lm.id = lm_inrange[j].id;
					lm.x  = lm_inrange[j].x;
	                lm.y  = lm_inrange[j].y;
					break;
				}
        	}

            // calc. weight (Multivariate Gaussian)
            if (lm.id == xobs.id) {

                norm        = (1/(2 * M_PI * sig_x * sig_y));
                exponent    = pow(xobs.x - lm.x,2)/(2 * pow(sig_x,2)) +
                              pow(xobs.y - lm.y,2)/(2 * pow(sig_y,2));
                particles[i].weight *= (norm * exp(-exponent));

                // cout << "+DEBUG+ current obs W: " << (norm * exp(-exponent)) << endl;
                cout << "   xobs " << k << ", " << "xobs id: " << setw(3) << xobs.id 
                     << ", current acc. weight: " << particles[i].weight << endl;

            } else {
                
            }


		}
        // accumulate weight of all particles as normalizer
        weight_norm += particles[i].weight;

        // overwrite particle weight in weights list
        weights[i]  = particles[i].weight;

        cout << "[Info] P" << i << ", particle.id: " << particles[i].id 
             << ", raw weight: " << weights[i] << endl;
    }

    cout << "\n[Info] = Normalizing Weights = normalizer: " << weight_norm << endl;

	// normalizing weights
	for (int w=0; w<weights.size(); w++) {
        weights[w] = weights[w] / weight_norm;

        cout << "[Info] P" << w << 
            ", raw: " << particles[w].weight <<
            ", normalized: "   << weights[w] << endl;

        particles[w].weight = weights[w];
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int> dist_particles(weights.begin(), weights.end());

    vector<Particle> resampled_particles;

    for (int i = 0; i < num_particles; i++) {
        resampled_particles.push_back(particles[dist_particles(gen)]);
    }

    particles = resampled_particles;
}
/*
Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}
*/
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

void ParticleFilter::print_particles(std::string identifier) 
{
    cout << "----------------------------------------------------------" << endl;
    cout << "[Info]: " << identifier << endl;
    for (int i=0; i<num_particles; i++) {
        cout << "P"             << i
             << ", id: "        << particles[i].id
             << ", x: "         << particles[i].x
             << ", y: "         << particles[i].y
             << ", theta: "     << particles[i].theta
             << ", weight: "    << particles[i].weight << endl;
    }
    cout << "----------------------------------------------------------" << endl;
}

double ParticleFilter::calc_dist(double x1, double y1, double x2, double y2)
{
    //calculating distance by euclidean formula
    return sqrt(pow(x1-x2,2) + pow(y1-y2,2));
}
