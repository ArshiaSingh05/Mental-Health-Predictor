import React from "react";
import { team } from "../../dummydata";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub, faLinkedin, faXTwitter, faInstagram } from "@fortawesome/free-brands-svg-icons";

const TeamCard = () => {
  return (
    <>
      {team.map((val, index) => (
        <div key={index} className="items shadow">
          <div className="img">
            <img src={val.cover} alt="" />
            <div className="overlay">
              <a href={val.github} target="_blank" rel="noopener noreferrer" className="icon">
                <FontAwesomeIcon icon={faGithub} />
              </a>
              <a href={val.linkedin} target="_blank" rel="noopener noreferrer" className="icon">
                <FontAwesomeIcon icon={faLinkedin} />
              </a>
              <a href={val.twitter} target="_blank" rel="noopener noreferrer" className="icon">
                <FontAwesomeIcon icon={faXTwitter} />
              </a>
              <a href={val.instagram} target="_blank" rel="noopener noreferrer" className="icon">
                <FontAwesomeIcon icon={faInstagram} />
              </a>
            </div>
          </div>
          <div className="details">
            <h2>{val.name}</h2>
            <p>{val.work}</p>
          </div>
        </div>
      ))}
    </>
  );
};

export default TeamCard;
