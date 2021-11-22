#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace std;
using namespace filesystem;
using namespace rapidjson;

inline double getSeconds(string& l){
  return stod(l.substr(0,l.size()-1));
}

string getDocId(string& f,int time){
  ostringstream oss;
  oss << "spotify:episode:" << f.substr(0,f.size()-5) << "_" << time << ".0";
  return oss.str();
}

Value getDoc(string& f, int time, string& content, Document::AllocatorType& allocator){
  Value obj(kObjectType);
  Value val(kObjectType);
  string id = getDocId(f,time);
  // cout << id << " " << content << endl;
  val.SetString(id.c_str(), static_cast<SizeType>(id.length()), allocator);
  obj.AddMember("id", val, allocator);
  val.SetString(content.c_str(), static_cast<SizeType>(content.length()), allocator);
  obj.AddMember("contents", val, allocator);
  return obj;
}

int saveContent(ostringstream& segmentOss,double wEt, int endTime, Document& outDocument, string& filename,Document::AllocatorType& allocator){
  string content = segmentOss.str();
  outDocument.PushBack(getDoc(filename,endTime-60,content,allocator),allocator);
  // cout << wEt << " " << endTime << endl;
  while(wEt>=endTime){
    endTime+=60;
  }
  // cout << wEt << " " << endTime << endl;
  segmentOss.str("");
  segmentOss.clear();
  return endTime;
}

void index_path(string filepath, string filename)
{
    std::ostringstream sstream;
    std::ifstream fs(filepath);
    sstream << fs.rdbuf();
    const std::string str(sstream.str());
    const char* ptr = str.c_str();
    Document document;
    document.Parse(ptr);
    Value& results = document["results"];
    assert(results.IsArray());

    int endTime = 60;
    ostringstream segmentOss;
    Document outDocument;
    outDocument.SetArray();
    Document::AllocatorType& allocator = outDocument.GetAllocator();
    for (SizeType i =  results.Size()-1; i < results.Size(); i++) {
      Value& results0 = results[i];
      if(results[0].HasMember("alternatives")){
        Value& results_alts = results0["alternatives"];
        Value& results_lists = results_alts[0];
        // cout << results_lists.MemberCount();
        if(results_lists.HasMember("words")){
          Value& results1 = results_lists["words"];
          for (SizeType j = 0; j < results1.Size(); j++) {
            string wordEndTime = results1[j]["endTime"].GetString();
            double wEt = getSeconds(wordEndTime);
            if(wEt < endTime){
              segmentOss << results1[j]["word"].GetString() << " ";
            }
            else{
              if(segmentOss.tellp()!=0)
                endTime = saveContent(segmentOss,wEt, endTime, outDocument, filename, allocator);
                segmentOss << results1[j]["word"].GetString() << " ";
            }
          }
        }
      }
      // cout << "AAAA" <<  segmentOss.str().size() << endl;
    }
  if(segmentOss.tellp()!=0)
    saveContent(segmentOss, 0, endTime, outDocument, filename, allocator);
  ofstream ofs("output/" + filename);
  OStreamWrapper osw(ofs);  
  Writer<OStreamWrapper> writer(osw);
  outDocument.Accept(writer);

}

int main()
{   
  // index_path("podcasts-no-audio-13GB/spotify-podcasts-2020/podcasts-transcripts/7/0/show_700JlWgEa3r2WfiB1VBeMU/0QcvUa7mAxvp5o4FYVhBcl.json","0QcvUa7mAxvp5o4FYVhBcl.json");
  for (recursive_directory_iterator i("podcasts-no-audio-13GB/spotify-podcasts-2020"), end; i != end; ++i) {
    if (!is_directory(i->path())){
      cout << i->path().filename() << "\n";
      index_path(i->path().native(),i->path().filename());
    }
  }
  return 0;
}
