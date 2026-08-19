#ifndef LANDMARK_MAP_H_
#define LANDMARK_MAP_H_

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

/****************************************************************************
 * @brief Landmark
 * A single map landmark. Stores its stable identifier, estimated global
 * position and the (standard-deviation) uncertainty of that estimate, plus a
 * flag recording whether the landmark has actually been observed yet.
 * ***************************************************************************/
struct Landmark {
  int    id       = -1;     // Stable landmark identifier
  double x        = 0.0;    // Global x position
  double y        = 0.0;    // Global y position
  double std_x    = 0.0;    // Standard deviation of x
  double std_y    = 0.0;    // Standard deviation of y
  bool   observed = false;  // Whether the landmark has been observed

  Landmark()      = default;
  Landmark(int id_, double x_, double y_, double std_x_ = 0.0,
           double std_y_ = 0.0, bool observed_ = true)
      : id(id_),
        x(x_),
        y(y_),
        std_x(std_x_),
        std_y(std_y_),
        observed(observed_) {}

  Eigen::Vector2d position() const { return Eigen::Vector2d(x, y); }
};

/****************************************************************************
 * @brief LandmarkMap
 * Container that manages the landmarks discovered during SLAM. Landmarks are
 * keyed by their id, so adding a landmark with an existing id updates it
 * (upsert semantics). The container can be serialised to / from a simple CSV
 * file and can be synchronised with an EKFSLAM state vector, which allows the
 * map to be saved after a run and reloaded to seed a later one.
 * ***************************************************************************/
class LandmarkMap {
 public:
  // ------------------------------------------------------------------ add
  // Insert a landmark or overwrite the existing one with the same id.
  void addLandmark(const Landmark& landmark) {
    landmarks_[landmark.id] = landmark;
  }

  void addLandmark(int id, double x, double y, double std_x = 0.0,
                   double std_y = 0.0, bool observed = true) {
    landmarks_[id] = Landmark(id, x, y, std_x, std_y, observed);
  }

  // --------------------------------------------------------------- remove
  // Returns true if a landmark with the given id existed and was removed.
  bool removeLandmark(int id) { return landmarks_.erase(id) > 0; }

  void clear() { landmarks_.clear(); }

  // -------------------------------------------------------------- retrieve
  bool contains(int id) const {
    return landmarks_.find(id) != landmarks_.end();
  }

  // Copy the landmark into out. Returns false if it does not exist.
  bool getLandmark(int id, Landmark& out) const {
    auto it = landmarks_.find(id);
    if (it == landmarks_.end()) {
      return false;
    }
    out = it->second;
    return true;
  }

  // Non-owning pointer to the stored landmark, or nullptr if absent.
  const Landmark* find(int id) const {
    auto it = landmarks_.find(id);
    return it == landmarks_.end() ? nullptr : &it->second;
  }

  // Snapshot of every landmark, ordered by id.
  std::vector<Landmark> getAll() const {
    std::vector<Landmark> result;
    result.reserve(landmarks_.size());
    for (const auto& entry : landmarks_) {
      result.push_back(entry.second);
    }
    return result;
  }

  std::vector<int> ids() const {
    std::vector<int> result;
    result.reserve(landmarks_.size());
    for (const auto& entry : landmarks_) {
      result.push_back(entry.first);
    }
    return result;
  }

  size_t size() const { return landmarks_.size(); }
  bool   empty() const { return landmarks_.empty(); }

  // ---------------------------------------------------------- serialization
  /**************************************************************************
   * @brief save
   * Write all landmarks to a CSV file (header + one row per landmark).
   * @param path destination file path
   * @return true on success, false if the file could not be written
   * ************************************************************************/
  bool   save(const std::string& path) const {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
      return false;
    }
    file << "id,x,y,std_x,std_y,observed\n";
    for (const auto& entry : landmarks_) {
      const Landmark& lm = entry.second;
      file << lm.id << "," << lm.x << "," << lm.y << "," << lm.std_x << ","
           << lm.std_y << "," << (lm.observed ? 1 : 0) << "\n";
    }
    return static_cast<bool>(file);
  }

  /**************************************************************************
   * @brief load
   * Replace the current contents with landmarks read from a CSV file that was
   * produced by save().
   * @param path source file path
   * @return true on success, false if the file could not be opened
   * ************************************************************************/
  bool load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      return false;
    }
    landmarks_.clear();

    std::string line;
    std::getline(file, line);  // discard header row
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      std::stringstream ss(line);
      std::string       token;
      Landmark          lm;

      if (!std::getline(ss, token, ',')) continue;
      lm.id = std::stoi(token);
      if (!std::getline(ss, token, ',')) continue;
      lm.x = std::stod(token);
      if (!std::getline(ss, token, ',')) continue;
      lm.y = std::stod(token);
      if (!std::getline(ss, token, ',')) continue;
      lm.std_x = std::stod(token);
      if (!std::getline(ss, token, ',')) continue;
      lm.std_y = std::stod(token);
      if (std::getline(ss, token, ',')) {
        lm.observed = (std::stoi(token) != 0);
      }

      landmarks_[lm.id] = lm;
    }
    return true;
  }

 private:
  std::map<int, Landmark> landmarks_;
};

#endif  // LANDMARK_MAP_H_
