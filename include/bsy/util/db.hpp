#pragma once


#include "bsy/common.hpp"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"


namespace bsy{
namespace db{

    enum Mode { READ, WRITE, NEW };

    class DB {
        public:
            DB(){}
            virtual ~DB(）=0；
            virtual void Open(const string & source, const Mode & mode) = 0;
            virtual void CursorSeekFirst() = 0;
            virtual void CursorNext() = 0;
            virtual string GetCurrentKey() = 0;
            virtual string GetCurrentValue() = 0;
            virtual bool CursorValid() = 0;
            virtual void Put(const string& key, const string& value) = 0;
            virtual void Commit() = 0;
            FORBID_COPY_AND_ASSIGN(DataBlock)
    };



    class LevelDB public: DB {
        public:
            LevelDB(): db_(NULL) {}
            virtual ~LevelDB(){
                if (db_ != NULL) {
                  delete db_;
                  db_ = NULL;
                }
            }

            virtual Open(const string & source, const Mode & mode) {
                leveldb::Options options;
                options.block_size = 65536;
                options.write_buffer_size = 268435456;
                options.max_open_files = 100;
                options.error_if_exists = mode == NEW;
                options.create_if_missing = mode != READ;
                leveldb::Status status = leveldb::DB::Open(options, source, &db_);
                CHECK(status.ok()) << "Failed to open leveldb " << source
                                 << std::endl << status.ToString();
                LOG(INFO) << "Opened leveldb " << source;
            }

            virtual CursorSeekFirst() {
                iter_->SeekToFirst();
            }

            virtual CursorNext() {
                if(!iter_){
                    iter_ = db_->NewIterator(leveldb::ReadOptions());
                    this->CursorSeekFirst();
                }
                iter_->Next();
            }

            virtual string GetCurrentKey() {
                return iter_->key().ToString();
            }

            virtual string GetCurrentValue() {
                return iter_->value().ToString();
            }

            virtual bool CursorValid() {
                return iter_->Valid();
            }

            virtual void Put(const string& key, const string& value){
                batch_.Put(key, value);
            }

            virtual void Commit() {
                leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
                CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
            }



        private:
            leveldb::DB* db_;
            leveldb::Iterator* iter_;
            leveldb::WriteBatch batch_;
    }

    DB * GetDB(const string& db_type){
        if (backend == "leveldb") {
            #ifdef USE_LEVELDB
            return new LevelDB();
            #endif
        }
        LOG(FATAL) << "Unknown database backend";
        return NULL;
    }

}// end namespace db
}// end namespace bsy